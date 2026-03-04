import os
import json
import shutil
import logging
import itertools
from pathlib import Path
from typing import Callable, Dict, Union, List

import hydra
import torch
import datasets
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets, load_dataset


logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        data: datasets.Dataset,
        stats: Dict,
        tokenizer: Callable,
    ):
        super().__init__()

        self.config = config
        self.data = data
        self.stats = stats
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Dict[str, list]:
        data = self.data[index]

        return {
            "action_tokens": data["action_tokens"],
            "precept_tokens": data["precept_tokens"],
            "total_action_tokens": sum(data["action_tokens_length"]),
            "total_precept_tokens": sum(data["precept_tokens_length"]),
        }

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build_dataset(
        cls,
        config: DictConfig,
    ) -> "TrajectoryDataset":
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

        datasets = cls.conform_example_format(
            datasets,
            null_symbol="~",
        )

        dataset = concatenate_datasets(datasets)

        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="actions",
            dst_columns={
                "tokens": "action_tokens",
            },
        )

        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="precepts",
            dst_columns={
                "tokens": "precept_tokens",
            },
        )

        dataset = cls.filter_length(
            dataset,
            max_action_len=config.max_action_len,
            max_precept_len=config.max_precept_len,
        )

        dataset = cls.pad_and_mask(
            dataset,
            pad=config.pad,
            src_column="action_tokens",
            dst_column="action_tokens_mask",
            max_length=config.max_action_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.pad_and_mask(
            dataset,
            pad=config.pad,
            src_column="precept_tokens",
            dst_column="precept_tokens_mask",
            max_length=config.max_precept_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.flatten_tajectories(dataset)

        dataset = dataset.remove_columns(["actions", "precepts"])

        dataset = dataset.flatten_indices()

        stats = cls.get_stats(dataset)

        return cls(
            config=config,
            data=dataset,
            stats=stats,
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

                    def completeness(example):
                        if example["response_quality_assessment"] != "":

                            return json.loads(example["response_quality_assessment"])[
                                "completeness"
                            ]["score"]

                        else:
                            return 0

                    dataset = dataset.filter(lambda example: completeness(example) >= 4)

                    dataset = dataset.filter(
                        lambda example: example["subset_name"]
                        in ["single-turn-diversify", "single-turn-original"]
                    )

                    dataset_list.append(dataset)
            else:
                raise NotImplementedError

        return dataset_list

    @staticmethod
    def conform_example_format(
        datasets: List[datasets.Dataset],
        null_symbol: str,
    ) -> List[datasets.Dataset]:

        def conform_toucan(examples):
            messages = [
                json.loads(examples["messages"][i])
                for i in range(len(examples["messages"]))
            ]
            actions = []
            precepts = []
            example_ids = []
            traj_lens = []

            for idx, msg_trace in enumerate(messages):
                msg_idx = 0
                actions_traj = []
                precept_traj = []

                # Skip traces that don't start with system → user
                if (
                    len(msg_trace) < 2
                    or msg_trace[0]["role"] != "system"
                    or msg_trace[1]["role"] != "user"
                ):
                    actions.append([])
                    precepts.append([])
                    traj_lens.append([])
                    example_ids.append([])
                    continue

                while msg_idx < len(msg_trace):

                    # System prompt
                    if msg_idx == 0:
                        precept_traj.append(msg_trace[msg_idx]["content"])
                        actions_traj.append(null_symbol)
                        msg_idx += 1

                    # User prompt
                    elif msg_idx == 1:
                        precept_traj.append(msg_trace[msg_idx]["content"])

                        if msg_idx + 1 >= len(msg_trace):
                            actions_traj.append(null_symbol)

                        msg_idx += 1

                    # All later steps.
                    else:

                        if len(actions_traj) < len(precept_traj):

                            assert msg_trace[msg_idx]["role"] == "assistant"

                            if msg_trace[msg_idx]["content"] != "":
                                actions_traj.append(msg_trace[msg_idx]["content"])
                                msg_idx += 1
                            elif (
                                "function_call" in msg_trace[msg_idx]
                                and msg_trace[msg_idx]["function_call"] != ""
                            ):
                                func_calls = []
                                while (
                                    msg_idx < len(msg_trace)
                                    and msg_trace[msg_idx]["role"] == "assistant"
                                    and "function_call" in msg_trace[msg_idx]
                                    and msg_trace[msg_idx]["function_call"] != ""
                                ):
                                    func_calls.append(
                                        json.dumps(msg_trace[msg_idx]["function_call"])
                                    )
                                    msg_idx += 1
                                actions_traj.append("".join(func_calls))
                            else:
                                # Empty content, no function_call — skip
                                msg_idx += 1

                        else:  # len(precepts_traj) == len(action_traj)

                            # Add precepts.
                            if msg_trace[msg_idx]["role"] == "function":
                                func_resps = []
                                while (
                                    msg_idx < len(msg_trace)
                                    and msg_trace[msg_idx]["role"] == "function"
                                ):
                                    func_resps.append(msg_trace[msg_idx]["content"])
                                    msg_idx += 1
                                precept_traj.append("".join(func_resps))

                                if msg_idx >= len(msg_trace):
                                    actions_traj.append(null_symbol)

                            # Add actions, infill precepts.
                            elif msg_trace[msg_idx]["role"] == "assistant":
                                while (
                                    msg_idx < len(msg_trace)
                                    and msg_trace[msg_idx]["role"] == "assistant"
                                ):
                                    if msg_trace[msg_idx]["content"] != "":
                                        actions_traj.append(
                                            msg_trace[msg_idx]["content"]
                                        )
                                        precept_traj.append(null_symbol)
                                        msg_idx += 1
                                    elif (
                                        "function_call" in msg_trace[msg_idx]
                                        and msg_trace[msg_idx]["function_call"] != ""
                                    ):
                                        func_calls = []
                                        while (
                                            msg_idx < len(msg_trace)
                                            and msg_trace[msg_idx]["role"]
                                            == "assistant"
                                            and "function_call" in msg_trace[msg_idx]
                                            and msg_trace[msg_idx]["function_call"]
                                            != ""
                                        ):
                                            func_calls.append(
                                                json.dumps(
                                                    msg_trace[msg_idx]["function_call"]
                                                )
                                            )
                                            msg_idx += 1
                                        actions_traj.append("".join(func_calls))
                                        precept_traj.append(null_symbol)
                                    else:
                                        # Empty content, no function_call — skip
                                        msg_idx += 1

                            # Follow-up user message — new precept.
                            elif msg_trace[msg_idx]["role"] == "user":
                                precept_traj.append(msg_trace[msg_idx]["content"])
                                msg_idx += 1

                                if msg_idx >= len(msg_trace):
                                    actions_traj.append(null_symbol)

                            else:
                                raise ValueError(
                                    f"Unexpected role '{msg_trace[msg_idx]['role']}' at msg_idx={msg_idx}"
                                )

                # Append terminal step: null precept, "done" action
                precept_traj.append(null_symbol)
                actions_traj.append("done")

                actions.append(actions_traj)
                precepts.append(precept_traj)
                traj_lens.append(len(actions_traj) * [len(actions_traj)])
                example_ids.append(len(actions_traj) * [examples["uuid"][idx]])

            examples["actions"] = list(itertools.chain.from_iterable(actions))
            examples["precepts"] = list(itertools.chain.from_iterable(precepts))
            examples["num_steps"] = list(itertools.chain.from_iterable(traj_lens))
            examples["example_ids"] = list(itertools.chain.from_iterable(example_ids))

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

            tokenized = []

            for sequence_list in examples[src_column]:
                tokenized.append(
                    tokenizer(
                        sequence_list,
                        add_special_tokens=True,
                        **kwargs,
                    )
                )

            eos_id = tokenizer.eos_token_id
            examples[dst_columns["tokens"]] = [
                item["input_ids"] + [eos_id] for item in tokenized
            ]

            if "mask" in dst_columns:
                examples[dst_columns["mask"]] = [
                    item["attention_mask"] for item in tokenized
                ]

            return examples

        return dataset.map(tokenize_fn, batched=True, cache_file_name=None)

    @staticmethod
    def filter_length(
        dataset: datasets.Dataset,
        max_action_len: int,
        max_precept_len: int,
    ) -> datasets.Dataset:

        os.makedirs("temp_filtered", exist_ok=True)

        df = pl.from_arrow(dataset.data.table).lazy()

        (
            df.with_columns(
                pl.col("action_tokens").list.len().alias("action_token_length"),
                pl.col("precept_tokens").list.len().alias("precept_token_length"),
            )
            .group_by("example_ids")
            .agg(
                pl.col("action_token_length").max().alias("max_action_length"),
                pl.col("precept_token_length").max().alias("max_precept_length"),
            )
            .filter(
                pl.col("max_action_length") <= max_action_len,
                pl.col("max_precept_length") <= max_precept_len,
            )
            .select("example_ids")
            .sink_parquet("temp_filtered/valid_ids.parquet")
        )

        valid_ids = pl.scan_parquet("temp_filtered/valid_ids.parquet")

        (
            df.join(valid_ids, on="example_ids", how="semi").sink_parquet(
                "temp_filtered/filtered_data.parquet"
            )
        )

        filtered_dataset = load_dataset(
            "parquet",
            data_files="temp_filtered/filtered_data.parquet",
            split="train",
        )

        shutil.rmtree("temp_filtered")

        return filtered_dataset

    @staticmethod
    def pad_and_mask(
        dataset: datasets.Dataset,
        pad: bool,
        src_column: str,
        dst_column: str,
        max_length: int,
        pad_token_id: int,
    ) -> datasets.Dataset:

        def _pad_and_mask(examples):
            lengths = []
            input_ids = []
            attention_masks = []

            for tokens in examples[src_column]:

                length = len(tokens)
                lengths.append(length)

                if pad:
                    input_ids.append(tokens + [pad_token_id] * (max_length - length))
                    attention_masks.append([1] * length + [0] * (max_length - length))
                else:
                    input_ids.append(tokens)
                    attention_masks.append([1])

            return {
                src_column: input_ids,
                dst_column: attention_masks,
                src_column + "_length": lengths,
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
    def flatten_tajectories(dataset: datasets.Dataset) -> datasets.Dataset:

        os.makedirs("temp_flattened", exist_ok=True)

        (
            pl.from_arrow(dataset.data.table)
            .lazy()
            .group_by(
                "example_ids",
                maintain_order=True,
            )
            .agg(pl.all())
            .with_columns(
                pl.col("num_steps").list.first(),
                c,
            )
            .sink_parquet("temp_flattened/flat_trajectory.parquet")
        )

        dataset = load_dataset(
            "parquet",
            data_files="temp_flattened/flat_trajectory.parquet",
            split="train",
        )

        shutil.rmtree("temp_flattened")

        return dataset

    @staticmethod
    def get_stats(dataset: datasets.Dataset) -> datasets.Dataset:
        action_tokens_length = np.concatenate(
            [np.array(i) for i in dataset["action_tokens_length"]]
        )
        precept_tokens_length = np.concatenate(
            [np.array(i) for i in dataset["precept_tokens_length"]]
        )
        trajectory_lengths = np.array(dataset["num_steps"])

        # Exclude null sequences
        action_tokens_length = action_tokens_length[action_tokens_length > 2]
        precept_tokens_length = precept_tokens_length[precept_tokens_length > 2]

        action_quantiles = np.percentile(action_tokens_length, [25, 50, 75, 95, 99])
        precept_quantiles = np.percentile(precept_tokens_length, [25, 50, 75, 95, 99])
        trajectory_quantiles = np.percentile(trajectory_lengths, [25, 50, 75, 95, 99])

        mapping = {
            "action": {
                "lengths": action_tokens_length,
                "quantiles": action_quantiles,
            },
            "precept": {
                "lengths": precept_tokens_length,
                "quantiles": precept_quantiles,
            },
            "trajectory": {
                "lengths": trajectory_lengths,
                "quantiles": trajectory_quantiles,
            },
        }

        stats = {}

        for key, val in mapping.items():
            stats[key] = {
                "mean": float(val["lengths"].mean()),
                "std": float(val["lengths"].std()),
                "min": int(val["lengths"].min()),
                "max": int(val["lengths"].max()),
                "q25": float(val["quantiles"][0]),
                "q50": float(val["quantiles"][1]),
                "q75": float(val["quantiles"][2]),
                "q95": float(val["quantiles"][3]),
                "q99": float(val["quantiles"][4]),
            }

        return stats

    def save(self, tgt_dir: Union[str, Path]) -> None:
        tgt_dir = Path(tgt_dir)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        # Save config.
        with open(tgt_dir / "config.yaml", "w") as f:
            OmegaConf.save(self.config, f)

        # Save data
        self.data.save_to_disk(str(tgt_dir / "data"))

        # Save stats.
        with open(tgt_dir / "stats.json", "w") as f:
            json.dump(self.stats, f)

    @classmethod
    def load(cls, src_dir: Union[str, Path]) -> "TrajectoryDataset":
        src_dir = Path(src_dir)

        # Load config
        with open(src_dir / "config.yaml", "r") as f:
            config = OmegaConf.load(f)

        # Load data
        data = datasets.load_from_disk(str(src_dir / "data"))

        # Load stats.
        with open(src_dir / "stats.json", "r") as f:
            stats = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        return cls(
            data=data,
            config=config,
            stats=stats,
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

    dataset = TrajectoryDataset.build_dataset(config=cfg.dataset)

    dataset.save(
        tgt_dir=Path(
            cfg.training.cache_dir,
            cfg.training.dataset_name,
        )
    )


if __name__ == "__main__":
    main()
