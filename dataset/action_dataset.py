import os
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Union, List

import datasets
from datasets import concatenate_datasets, load_dataset
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import polars as pl
from torch.utils.data import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class ActionDataset(Dataset):

    def __init__(self, config: DictConfig, data: datasets.Dataset, tokenizer: Callable):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return {
            "input_ids": self.data["action_tokens"][index],
            "attention_mask": self.data["attention_mask"][index],
        }

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build_dataset(
        cls,
        config: DictConfig,
    ) -> "ActionDataset":
        """Factory function."""

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Ensure tokenizer has pad_token
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must have pad token")

        datasets = cls.download_datasets(
            dataset_names=config.dataset_names,
            debug=config.debug,
        )

        datasets = cls.conform_example_format(datasets)

        dataset = concatenate_datasets(datasets)

        dataset = dataset.filter(lambda x: len(x["actions"]) > 0).flatten_indices()

        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="actions",
            dst_columns={
                "tokens": "action_tokens",
            },
        )

        logging.info("Tokenized!")

        dataset = cls.filter_length(dataset, config.max_action_len)

        dataset = cls.pad_and_mask(
            dataset,
            max_length=config.max_action_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.create_null_sequences(
            dataset,
            max_len=config.max_action_len,
            null_ratio=config.null_ratio,
            null_token_id=tokenizer.pad_token_id,
        )

        dataset = dataset.flatten_indices()

        dataset.set_format(type="torch", columns=["action_tokens", "attention_mask"])

        dataset = dataset.shuffle(seed=42)

        return cls(
            config=config,
            data=dataset,
            tokenizer=tokenizer,
        )

    @staticmethod
    def download_datasets(
        dataset_names: Union[str, List[str]],
        debug: bool,
    ) -> List[datasets.Dataset]:

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        dataset_list = []

        for dataset_name in dataset_names:
            if dataset_name == "Agent-Ark/Toucan-1.5M":

                if debug:
                    splits = ["Kimi-K2"]
                else:
                    splits = ["Kimi-K2", "OSS", "Qwen3"]  # , SFT

                for split in splits:

                    dataset = load_dataset(
                        dataset_name,
                        split,
                    )["train"]

                    if debug:
                        dataset = dataset.select(range(5 * 32))

                    dataset_list.append(dataset)
            else:
                raise NotImplementedError

        return dataset_list

    @staticmethod
    def conform_example_format(
        datasets: List[datasets.Dataset],
    ) -> List[datasets.Dataset]:

        def conform_toucan(examples):
            messages = [
                json.loads(examples["messages"][i])
                for i in range(len(examples["messages"]))
            ]

            actions = []
            ids = []

            for idx, msg_trace in enumerate(messages):
                for msg in msg_trace:
                    if msg["role"] == "assistant":
                        actions.append(msg["content"])
                        ids.append(examples["uuid"][idx])

            examples["actions"] = actions
            examples["id"] = ids

            return examples

        conformed_datasets = []

        for dataset in datasets:
            if dataset._info.dataset_name == "toucan-1.5_m":
                conformed_datasets.append(
                    dataset.map(
                        conform_toucan,
                        batched=True,
                        cache_file_name=None,
                        remove_columns=[
                            "uuid",
                            "question",
                            "subset_name",
                            "messages",
                            "available_tools",
                            "target_tools",
                            "question_quality_assessment",
                            "response_quality_assessment",
                            "metadata",
                        ],
                        desc="Conforming dataset examples...",
                    )
                )
            else:
                raise NotImplementedError

        return conformed_datasets

    @staticmethod
    def _tokenize(
        dataset: datasets.Dataset,
        tokenizer: Callable,
        src_column: str,
        dst_columns: Dict[str, str],
        **kwargs,
    ) -> datasets.Dataset:

        def tokenize_fn(examples):

            tokenized = tokenizer(
                examples[src_column],
                add_special_tokens=False,
                **kwargs,
            )

            examples[dst_columns["tokens"]] = tokenized["input_ids"]

            if "mask" in dst_columns:
                examples[dst_columns["mask"]] = tokenized["attention_mask"]

            return examples

        return dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=8,
            desc="Tokenizing...",
        )

    @staticmethod
    def filter_length(
        dataset: datasets.Dataset,
        max_action_len: int,
    ) -> datasets.Dataset:

        os.makedirs("temp_filtered", exist_ok=True)

        df = pl.from_arrow(dataset.data.table).lazy()

        # Find valid IDs and write to disk
        (
            df.with_columns(pl.col("action_tokens").list.len().alias("token_length"))
            .group_by("id")
            .agg(pl.col("token_length").max().alias("max_length"))
            .filter(pl.col("max_length") <= max_action_len)
            .select("id")
            .sink_parquet("temp_filtered/valid_ids.parquet")
        )

        # Filter and write to disk
        valid_ids = pl.scan_parquet("temp_filtered/valid_ids.parquet")

        (
            df.join(valid_ids, on="id", how="semi").sink_parquet(
                "temp_filtered/filtered_data.parquet"
            )
        )

        filtered_dataset = load_dataset(
            "parquet",
            data_files="temp_filtered/filtered_data.parquet",
            split="train",
        )

        return filtered_dataset

    @staticmethod
    def pad_and_mask(
        dataset: datasets.Dataset,
        max_length: int,
        pad_token_id: int,
    ) -> datasets.Dataset:

        def _pad_and_mask(examples):
            input_ids = []
            attention_masks = []

            for tokens in examples["action_tokens"]:

                length = len(tokens)
                input_ids.append(tokens + [pad_token_id] * (max_length - length))
                attention_masks.append([1] * length + [0] * (max_length - length))

            return {
                "action_tokens": input_ids,
                "attention_mask": attention_masks,
                "token_length": [sum(mask) for mask in attention_masks],
            }

        return dataset.map(
            _pad_and_mask,
            batched=True,
            batch_size=1000,
            num_proc=8,
            remove_columns=None,
            desc="Padding and masking...",
        )

    @staticmethod
    def create_null_sequences(
        dataset: datasets.Dataset,
        max_len: int,
        null_ratio: float,
        null_token_id: int,
    ) -> datasets.Dataset:

        logging.info("Creating null sequences...")

        num_nulls = int(null_ratio * len(dataset))

        null_data = {
            "id": [""] * num_nulls,
            "actions": [""] * num_nulls,
            "action_tokens": [[null_token_id] * max_len] * num_nulls,
            "attention_mask": [[1] * max_len] * num_nulls,
            "token_length": [0] * num_nulls,
        }

        null_dataset = datasets.Dataset.from_dict(
            null_data,
            features=dataset.features,
        )

        return concatenate_datasets([dataset, null_dataset])

    def get_stats(self) -> Dict:

        token_lengths = np.array(self.data["token_length"])

        # Exclude null sequences
        token_lengths = token_lengths[token_lengths > 0]

        quantiles = np.percentile(token_lengths, [25, 50, 75, 95, 99])

        return {
            "mean": float(token_lengths.mean()),
            "std": float(token_lengths.std()),
            "min": int(token_lengths.min()),
            "max": int(token_lengths.max()),
            "q25": float(quantiles[0]),
            "q50": float(quantiles[1]),
            "q75": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
        }

    def save(self, tgt_dir: Union[str, Path]) -> None:
        tgt_dir = Path(tgt_dir)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        # Save config.
        with open(tgt_dir / "config.yaml", "w") as f:
            OmegaConf.save(self.config, f)

        # Save data
        self.data.save_to_disk(str(tgt_dir / "data"))

    @classmethod
    def load(cls, src_dir: Union[str, Path]) -> "ActionDataset":
        src_dir = Path(src_dir)

        # Load config
        with open(src_dir / "config.yaml", "r") as f:
            config = OmegaConf.load(f)

        # Load data
        data = datasets.load_from_disk(str(src_dir / "data"))

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        return cls(
            data=data,
            config=config,
            tokenizer=tokenizer,
        )


script_dir = Path(__file__).parent
if (script_dir / "conf").exists():
    # Container environment: config is next to script
    config_path = str(script_dir / "conf")
else:
    # Local environment: config is at project root
    project_root = script_dir.parent
    config_path = str(project_root / "conf")


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    dataset = ActionDataset.build_dataset(config=cfg.dataset)
    dataset.save(
        tgt_dir=Path(
            cfg.training.cache_dir,
            cfg.training.dataset_name,
        )
    )


if __name__ == "__main__":
    main()
