import logging
from typing import Callable, Dict, Optional, Union, List, Tuple


import datasets
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)

from prompts import PromptBuilder
from tools.tools import (
    ToolSignature,
    GOOGLE_SIGNATURE,
    WIKIPEDIA_SIGNATURE,
    TOOL_SIGNATURES,
    TOOL_DICT,
)


logger = logging.getLogger(__name__)


class PromptDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build_dataset(
        cls,
        model_name: str,
        debug: bool,
        dataset_names: Union[str, List[str]],
        max_observation_len: int,
        max_action_len: int,
        max_prompt_len: int,
        # prompt_batch_size: int,
        min_steps: int,
        max_steps: int,
        # max_actions: int,
        tool_ids: List[str],
        num_trajectories_per_example: int,
        # tools: List[Callable],
        tool_probs: Dict[str, float],
        # tokenizer: Callable,
    ) -> "PromptDataset":
        """Factory function."""

        # Set up language model.

        model_config = AutoConfig.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        # Set up tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Download base datasets.
        datasets = cls.download_datasets(dataset_names, debug=debug)

        # Conform dataset format.
        datasets = cls.conform_example_format(datasets)

        # Merge conformed datasets.
        if len(datasets) > 1:
            dataset = cls.merge_datasets(datasets)
        else:
            dataset = datasets[0]

        # Tokenize examples' questions.
        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="question_text",
            dst_columns={
                "tokens": "question_tokens",
                "mask": "question_attention_mask",
            },
            **{
                "padding": "max_length",
                "max_length": max_observation_len,
                "truncation": True,
            },
        )

        # Tokenize examples' answers.
        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="answer_text",
            dst_columns={
                "tokens": "answer_tokens",
                "mask": "answer_attention_mask",
            },
            **{
                "padding": "max_length",
                "max_length": max_action_len,
                "truncation": True,
            },
        )

        # Build tools.
        tool_registry = cls.build_tools(tool_ids)

        # Build prompt object
        prompt_builder = PromptBuilder(
            tools=tool_ids,
            tool_signatures={
                tool: sig for tool, sig in TOOL_SIGNATURES.items() if tool in tool_ids
            },
            tool_probs=tool_probs,
        )

        # Build base prompts
        dataset = cls.build_base_prompts(dataset, builder=prompt_builder)

        # Tokenize base prompts
        dataset = cls._tokenize(
            dataset,
            tokenizer,
            src_column="prompt_str",
            dst_columns={
                "tokens": "prompt_tokens",
                "mask": "prompt_attention_mask",
            },
            **{
                "padding": "max_length",
                "max_length": max_prompt_len,
                "truncation": True,
            },
        )

        # Add base id column.
        dataset = dataset.add_column("base_id", list(range(len(dataset))))

        # Add extra rows if multiple trajectories.
        if num_trajectories_per_example > 1:
            dataset = cls.duplicate_train_examples(
                dataset, n_duplicates=num_trajectories_per_example
            )

        # Randomly sample num_steps. Subtract one step to allow for final step to be answer action.
        dataset = cls.sample_steps(
            dataset, min_steps=min_steps, max_steps=max_steps - 1
        )

        # Sample which tools to use for each step.
        dataset = cls.sample_tool_trajectories(
            dataset, tool_ids=tool_ids, tool_probs=tool_probs
        )

        # Add step info/available APIs to prompt.
        dataset = cls.update_prompt(
            cls,
            dataset,
            builder=prompt_builder,
            tokenizer=tokenizer,
            step=0,
            max_len=max_prompt_len,
        )

        # Build up trajectories
        for step in range(max_steps):
            ### Start GPU Phase ###
            model_outputs = None

            ### End GPU Phase ###

            mock_calls = ["call1", "call2", "call3"]
            mock_resps = ["resp1", "resp2", "resp3"]

            # Decode y to string
            # Extract api calls from  y
            # Tokenize api calls, add to dataset["actions"]
            # Create and execute API calls
            # Update prompt

            # Add calls and responses to prompt.
            dataset = cls.update_prompt(
                cls,
                dataset,
                builder=prompt_builder,
                tokenizer=tokenizer,
                step=step + 1,
                max_len=max_prompt_len,
                calls=mock_calls,
                responses=mock_resps,
            )

            assert False

        #       For i in step up to this step

        #           For id in tool_ids

        #               tool_i_step_i = cat(tool_ids[id][call_tokens], tool_ids[id][response_tokens])

        #            step_i_tensor = cat([tool_0_step_i, tool_1_step_0])

        #       prompt = cat(base_prompt, step_0_tensor, step_1_tensor, ...)

        #       pad prompt to max_prompt_len

        #       add prompt to dictionary as current_prompt

        #   ### GPU ###

        #   Get API calls.

        #   For batch of examples in examples:

        #       batch = [batch_size, max_prompt_len]

        #       api_calls = model_wrapper(model, batch, num_samples, num_samples_per_example)

        #  ### CPU ###

        #  Post processing

        #  Filter the api_calls

        #      Filter out non conforming responses

        #      Check if there are num_samples_per_example remaining

        #      Do something if not

        #  Clean call

        #       Depending on tool, do some regex stuff to extract what you want

        #  Add call text to dictionary

        #  Batch tokenize the calls, add to dictionary

        #  invoke the APIs

        #      response = tool_id(call_text)

        # Clean response

        #       Regex stuff

    @staticmethod
    def download_datasets(
        dataset_names: Union[str, List[str]],
        debug: bool,
    ) -> List[datasets.Dataset]:

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        dataset_list = []

        for dataset_name in dataset_names:
            if dataset_name == "hotpot_qa":
                dataset = load_dataset(
                    dataset_name,
                    "distractor",
                    split="train+validation",
                )

                if debug:
                    dataset = dataset.select(range(64))

                dataset_list.append(dataset)
            else:
                raise NotImplementedError

        return dataset_list

    @staticmethod
    def conform_example_format(
        datasets: List[datasets.Dataset],
    ) -> List[datasets.Dataset]:

        def conform_hotpot_qa(examples):
            conformed_examples = {}
            conformed_examples["question_text"] = examples["question"]
            conformed_examples["answer_text"] = examples["answer"]

            return conformed_examples

        conformed_datasets = []

        for dataset in datasets:
            if dataset._info.dataset_name == "hotpot_qa":
                conformed_datasets.append(
                    dataset.map(
                        conform_hotpot_qa,
                        batched=True,
                        cache_file_name=None,
                        remove_columns=[
                            "id",
                            "question",
                            "answer",
                            "type",
                            "level",
                            "supporting_facts",
                            "context",
                        ],
                    )
                )
            else:
                raise NotImplementedError

        return conformed_datasets

    @staticmethod
    def merge_datasets(dataset_list: List[datasets.Dataset]) -> datasets.Dataset:
        pass

    @staticmethod
    def _tokenize(
        dataset: datasets.Dataset,
        tokenizer: Callable,
        src_column: str,
        dst_columns: Dict[str, str],
        **kwargs,
    ) -> datasets.Dataset:
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Tokenizer must have pad token")

        def tokenize_fn(examples):
            encoding = tokenizer(
                examples[src_column],
                **kwargs,
            )

            result = {}

            # Ensure token IDs are properly converted to lists of integers
            result[dst_columns["tokens"]] = [
                [int(token_id) for token_id in sequence]
                for sequence in encoding.input_ids
            ]

            if "mask" in dst_columns:
                result[dst_columns["mask"]] = [
                    [int(mask_val) for mask_val in sequence]
                    for sequence in encoding.attention_mask
                ]

            return result

        return dataset.map(tokenize_fn, batched=True, cache_file_name=None)

    @staticmethod
    def build_tools(tool_ids: List[str]) -> Dict[str, Callable]:

        registry = {}

        for id_ in tool_ids:
            try:
                id_ in TOOL_DICT.keys()
            except KeyError:
                raise ValueError(f"Key {id_} not found in tool dictionary")

            registry[id_] = TOOL_DICT[id_]

        return registry

    @staticmethod
    def duplicate_train_examples(
        dataset: datasets.Dataset,
        n_duplicates: int,
    ) -> datasets.Dataset:

        def duplicate_batch(batch):
            duplicated_batch = {
                column: [value for value in values for _ in range(n_duplicates)]
                for column, values in batch.items()
            }
            return duplicated_batch

        return dataset.map(
            duplicate_batch,
            batched=True,
            remove_columns=dataset.column_names,
            cache_file_name=None,
        )

    @staticmethod
    def sample_steps(
        dataset: datasets.Dataset, min_steps: int, max_steps: int
    ) -> datasets.Dataset:

        dataset = dataset.add_column(
            "num_steps",
            np.random.randint(low=min_steps, high=max_steps, size=len(dataset)),
        )

        return dataset

    @staticmethod
    def sample_tool_trajectories(
        dataset: datasets.Dataset, tool_ids: List[int], tool_probs: Dict[str, float]
    ) -> datasets.Dataset:

        def _sample_tool_trajectories(batch):
            trajectories = []
            trajectory_lengths = batch["num_steps"]

            # Pre-compute probabilities array
            probs_array = np.array([tool_probs[tool_id] for tool_id in tool_ids])

            for i in range(len(trajectory_lengths)):
                trajectory_length = trajectory_lengths[i]

                # Generate random matrix: [trajectory_length, num_tools]
                random_matrix = np.random.random((trajectory_length, len(tool_ids)))

                # Compare with probabilities to get binary usage matrix
                usage_matrix = (random_matrix < probs_array).astype(bool)

                # Convert to list of lists of tool names
                example_trajectory = []
                for step_usage in usage_matrix:
                    step_tools = [
                        tool_ids[j] for j, used in enumerate(step_usage) if used
                    ]
                    example_trajectory.append(step_tools)

                trajectories.append(example_trajectory)

            batch["tool_trajectory"] = trajectories

            return batch

        return dataset.map(
            _sample_tool_trajectories,
            batched=True,
            cache_file_name=None,
        )

    @staticmethod
    def build_base_prompts(
        dataset: datasets.Dataset, builder: PromptBuilder
    ) -> datasets.Dataset:

        def _build_base_prompt(batch):
            questions = batch["question_text"]
            batch["prompt_str"] = [builder.build_initial_prompt(q) for q in questions]
            return batch

        return dataset.map(_build_base_prompt, batched=True, cache_file_name=None)

    def update_prompt(
        self,
        dataset: datasets.Dataset,
        builder: PromptBuilder,
        tokenizer: Callable,
        step: int,
        max_len: int,
        calls: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
    ) -> datasets.Dataset:

        def _update(batch):
            if calls and responses:
                batch["prompt_str"] = [
                    builder.add_calls_and_responses(
                        batch["prompt_str"][i],
                        previous_calls=calls,
                        responses=responses,
                    )
                    for i in range(len(batch["prompt_str"]))
                ]

            batch["prompt_str"] = [
                builder.add_step_prompt(
                    batch["prompt_str"][i],
                    step_num=step,
                    available_tools=batch["tool_trajectory"][i][step],
                )
                for i in range(len(batch["prompt_str"]))
            ]

            return batch

        dataset = dataset.map(
            _update,
            batched=True,
            cache_file_name=None,
        )

        dataset = self._tokenize(
            dataset,
            tokenizer=tokenizer,
            src_column="prompt_str",
            dst_columns={
                "tokens": "prompt_tokens",
                "mask": "prompt_attention_mask",
            },
            **{
                "padding": "max_length",
                "max_length": max_len,
                "truncation": True,
            },
        )

        return dataset

    def construct_prompt(self):
        pass

    def generate_data_batch(self):
        pass

    def preprocess_api_call(self):
        pass

    def make_api_calls(self):
        pass

    def filter_api_responses(self):
        pass

    def filter_and_keep_only_first_api_response(self):
        pass

    def add_final_action(self):
        """Add <answer> as final API action"""
        pass

    def add_confounding_actions(self):
        pass

    def pad_sequence_batch(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    PromptDataset.build_dataset(
        model_name="gpt2",
        debug=True,
        dataset_names="hotpot_qa",
        max_observation_len=256,
        max_action_len=256,
        max_prompt_len=1024,
        min_steps=6,
        max_steps=10,
        tool_ids=["google", "wikipedia"],
        tool_probs={"google": 0.5, "wikipedia": 0.5},
        num_trajectories_per_example=2,
    )
