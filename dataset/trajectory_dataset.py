import os
import json
import shutil
import logging
import itertools
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List, Tuple


import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets, load_dataset
from einops import rearrange
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    GenerationConfig,
)


logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):

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

    def __getitem__(self, index):
        return super().__getitem__(index)

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

        datasets = cls.conform_example_format(datasets)

        dataset = concatenate_datasets(datasets)

        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="actions",
            dst_columns={
                "tokens": "action_tokens",
                "mask": "action_attention_mask",
            },
        )

        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="precepts",
            dst_columns={
                "tokens": "precept_tokens",
                "mask": "precept_attention_mask",
            },
        )

        dataset = cls.filter_length(
            dataset,
            max_action_len=config.max_action_len,
            max_precept_len=config.max_precept_len,
        )

        dataset = cls.add_null_steps(
            dataset,
            column="action_tokens",
            length=config.max_action_len,
            token_id=100,
        )

        dataset = cls.add_null_steps(
            dataset,
            column="precept_tokens",
            length=config.max_precept_len,
            token_id=100,
        )

        dataset = cls.pad_and_mask(
            dataset,
            src_column="action_tokens",
            dst_column="action_tokens_mask",
            max_length=config.max_action_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.pad_and_mask(
            dataset,
            src_column="precept_tokens",
            dst_column="precept_tokens_mask",
            max_length=config.max_precept_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.flatten_tajectories(dataset)

        dataset = cls.pad_trajectories(
            dataset,
            column="action_tokens",
            step_len=config.max_action_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        dataset = cls.pad_trajectories(
            dataset,
            column="precept_tokens",
            step_len=config.max_precept_len,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Add actions column.
        dataset = dataset.add_column("actions", column=len(dataset) * [[]])

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

                while msg_idx < len(msg_trace):

                    # System prompt
                    if msg_idx == 0:
                        assert msg_trace[msg_idx]["role"] == "system"
                        precept_traj.append(msg_trace[msg_idx]["content"])
                        actions_traj.append("")
                        msg_idx += 1

                    # User prompt
                    elif msg_idx == 1:
                        assert msg_trace[msg_idx]["role"] == "user"
                        precept_traj.append(msg_trace[msg_idx]["content"])

                        if msg_idx + 1 > len(msg_trace):
                            actions_traj.append("")

                        msg_idx += 1

                    # All later steps.
                    else:

                        if len(actions_traj) < len(precept_traj):

                            assert msg_trace[msg_idx]["role"] == "assistant"

                            if msg_trace[msg_idx]["content"] != "":
                                actions_traj.append(msg_trace[msg_idx]["content"])
                                msg_idx += 1
                            else:
                                func_calls = []
                                while (
                                    msg_trace[msg_idx]["role"] == "assistant"
                                    and msg_trace[msg_idx]["function_call"] != ""
                                ):
                                    func_calls.append(
                                        json.dumps(msg_trace[msg_idx]["function_call"])
                                    )
                                    msg_idx += 1
                                actions_traj.append("".join(func_calls))

                        else:  # len(precepts_traj) == len(action_traj)

                            # Add precepts.
                            if msg_trace[msg_idx]["role"] == "function":
                                func_resps = []
                                while msg_trace[msg_idx]["role"] == "function":
                                    func_resps.append(msg_trace[msg_idx]["content"])
                                    msg_idx += 1
                                precept_traj.append("".join(func_resps))

                                if msg_idx + 1 > len(msg_trace):
                                    actions_traj.append("")

                            # Add actions, infill precepts.
                            elif msg_trace[msg_idx]["role"] == "assistant":
                                while msg_trace[msg_idx]["role"] == "assistant":
                                    if msg_trace[msg_idx]["content"] != "":
                                        actions_traj.append(
                                            msg_trace[msg_idx]["content"]
                                        )
                                        precept_traj.append("")
                                        msg_idx += 1
                                    else:
                                        func_calls = []
                                        while (
                                            msg_trace[msg_idx]["role"] == "assistant"
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
                                        precept_traj.append("")

                            else:
                                raise ValueError("Multi-turn traces not supported yet.")

                    if len(actions_traj) == len(precept_traj):
                        example_ids.append(examples["uuid"][idx])

                actions.append(actions_traj)
                precepts.append(precept_traj)
                traj_lens.append(len(actions_traj) * [len(actions_traj)])

            examples["actions"] = list(itertools.chain.from_iterable(actions))
            examples["precepts"] = list(itertools.chain.from_iterable(precepts))
            examples["num_steps"] = list(itertools.chain.from_iterable(traj_lens))
            examples["example_ids"] = example_ids

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
                        add_special_tokens=False,
                        **kwargs,
                    )
                )

            examples[dst_columns["tokens"]] = [item["input_ids"] for item in tokenized]

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
    def add_null_steps(
        dataset: datasets.Dataset,
        column: str,
        length: int,
        token_id: int,
    ) -> datasets.Dataset:

        os.makedirs("temp_df", exist_ok=True)

        (
            pl.from_arrow(dataset.data.table)
            .lazy()
            .with_columns(
                pl.when(pl.col(column).list.len() == 0)
                .then(length * [token_id])
                .otherwise(pl.col(column))
                .name.keep()
            )
            .sink_parquet("temp_df/df.parquet")
        )

        dataset = load_dataset(
            "parquet",
            data_files="temp_df/df.parquet",
            split="train",
        )

        shutil.rmtree("temp_df")

        return dataset

    @staticmethod
    def pad_and_mask(
        dataset: datasets.Dataset,
        src_column: str,
        dst_column: str,
        max_length: int,
        pad_token_id: int,
    ) -> datasets.Dataset:

        def _pad_and_mask(examples):
            input_ids = []
            attention_masks = []

            for tokens in examples[src_column]:

                length = len(tokens)
                input_ids.append(tokens + [pad_token_id] * (max_length - length))
                attention_masks.append([1] * length + [0] * (max_length - length))

            return {
                src_column: input_ids,
                dst_column: attention_masks,
                src_column + "_length": [sum(mask) for mask in attention_masks],
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
            .with_columns(pl.col("num_steps").list.first())
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
    def pad_trajectories(
        dataset: datasets.Dataset,
        column: str,
        step_len: int,
        pad_token_id: int,
    ) -> datasets.Dataset:

        padded_trajectories = []
        attention_masks = []
        max_len = np.array(dataset["num_steps"]).max()

        def _pad_fn(examples):

            for trajectory in examples[column]:
                length = len(trajectory)
                padded_trajectories.append(
                    trajectory
                    + [step_len * [pad_token_id] for _ in range(max_len - length)]
                )
                attention_masks.append([1] * length + [0] * (max_len - length))

            return {
                column: padded_trajectories,
                column + "_tajectory_mask": attention_masks,
            }

        return dataset.map(
            _pad_fn,
            batched=True,
            batch_size=1000,
            num_proc=8,
            remove_columns=None,
            desc=f"Padding trajectories for {column}...",
        )

    def calculate_stats():
        pass

    def save(self):
        pass

    def load(self):
        pass


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
    # dataset.save(
    #     tgt_dir=Path(
    #         cfg.training.cache_dir,
    #         cfg.training.dataset_name,
    #     )
    # )


if __name__ == "__main__":
    main()
