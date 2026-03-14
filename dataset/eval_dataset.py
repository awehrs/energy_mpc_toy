import json
import logging
from pathlib import Path
from typing import Callable, Dict, Union, List

import datasets
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets, load_dataset


logger = logging.getLogger(__name__)


class EvalDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        data: datasets.Dataset,
        tokenizer: Callable,
    ):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Dict:
        row = self.data[index]
        return {
            "uuid": row["uuid"],
            "question": row["question"],
            "messages": json.loads(row["messages"]),
            "target_tools": [t.strip() for t in row["target_tools"].split(",") if t.strip()],
            "original_completeness": row["original_completeness"],
            "original_conciseness": row["original_conciseness"],
        }

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build_dataset(cls, config: DictConfig) -> "EvalDataset":
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must have pad token")

        raw_datasets = cls.download_datasets(
            dataset_names=config.dataset_names,
            max_completeness=config.max_completeness,
            debug=config.debug,
        )

        dataset = concatenate_datasets(raw_datasets)
        dataset = cls.filter_by_servers(dataset, list(config.server_whitelist))
        dataset = cls.extract_scores(dataset)

        keep = {"uuid", "question", "messages", "target_tools",
                "original_completeness", "original_conciseness"}
        drop = [col for col in dataset.column_names if col not in keep]
        dataset = dataset.remove_columns(drop)
        dataset = dataset.flatten_indices()

        return cls(config=config, data=dataset, tokenizer=tokenizer)

    @staticmethod
    def download_datasets(
        dataset_names: Union[str, List[str]],
        max_completeness: int,
        debug: bool,
    ) -> List[datasets.Dataset]:

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        dataset_list = []

        for dataset_name in dataset_names:
            if dataset_name == "Agent-Ark/Toucan-1.5M":

                splits = ["Kimi-K2"] if debug else ["Kimi-K2", "OSS", "Qwen3"]

                for split in splits:
                    dataset = load_dataset(dataset_name, split)["train"]

                    if debug:
                        dataset = dataset.select(range(5 * 32))

                    def get_completeness(example):
                        rqa = example["response_quality_assessment"]
                        if not rqa:
                            return 0
                        try:
                            return json.loads(rqa)["completeness"]["score"]
                        except (json.JSONDecodeError, KeyError):
                            return 0

                    dataset = dataset.filter(
                        lambda ex: 0 < get_completeness(ex) <= max_completeness
                    )
                    dataset = dataset.filter(
                        lambda ex: ex["subset_name"]
                        in ["single-turn-diversify", "single-turn-original"]
                    )

                    dataset_list.append(dataset)
            else:
                raise NotImplementedError

        return dataset_list

    @staticmethod
    def filter_by_servers(
        dataset: datasets.Dataset,
        server_whitelist: List[str],
    ) -> datasets.Dataset:
        """Keep only examples where all function calls use whitelisted servers."""

        def all_servers_allowed(example):
            try:
                messages = json.loads(example["messages"])
            except (json.JSONDecodeError, TypeError):
                return False

            for msg in messages:
                fc = msg.get("function_call")
                if not fc:
                    continue
                if isinstance(fc, str):
                    try:
                        fc = json.loads(fc)
                    except json.JSONDecodeError:
                        continue
                tool_name = fc.get("name", "") if isinstance(fc, dict) else ""
                if not tool_name:
                    continue
                if not any(tool_name.startswith(server + "-") for server in server_whitelist):
                    return False

            return True

        return dataset.filter(all_servers_allowed)

    @staticmethod
    def extract_scores(dataset: datasets.Dataset) -> datasets.Dataset:
        """Parse response_quality_assessment and add original score columns."""

        def parse_scores(examples):
            completeness_scores = []
            conciseness_scores = []

            for rqa in examples["response_quality_assessment"]:
                try:
                    parsed = json.loads(rqa)
                    completeness_scores.append(float(parsed["completeness"]["score"]))
                    conciseness_scores.append(float(parsed["conciseness"]["score"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    completeness_scores.append(0.0)
                    conciseness_scores.append(0.0)

            examples["original_completeness"] = completeness_scores
            examples["original_conciseness"] = conciseness_scores
            return examples

        return dataset.map(parse_scores, batched=True, cache_file_name=None)

    def save(self, tgt_dir: Union[str, Path]) -> None:
        tgt_dir = Path(tgt_dir)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        with open(tgt_dir / "config.yaml", "w") as f:
            OmegaConf.save(self.config, f)

        self.data.save_to_disk(str(tgt_dir / "data"))

    @classmethod
    def load(cls, src_dir: Union[str, Path]) -> "EvalDataset":
        src_dir = Path(src_dir)

        with open(src_dir / "config.yaml", "r") as f:
            config = OmegaConf.load(f)

        data = datasets.load_from_disk(str(src_dir / "data"))
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        return cls(config=config, data=data, tokenizer=tokenizer)
