import logging
from typing import Callable, Dict, Optional, Union, List, Tuple


import datasets
from datasets import load_dataset
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from vllm import LLM, SamplingParams, TokensPrompt
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    GenerationConfig,
)

from prompts import PromptBuilder
from tools.tools import (
    ToolSignature,
    TOOL_DICT,
    TOOL_SIGNATURES,
)


logger = logging.getLogger(__name__)


class PromptDataset(Dataset):

    def __init__(self, data: datasets.Dataset, memmap: Dict):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build_dataset(
        cls,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        debug: bool = True,
        dataset_names: Union[str, List[str]] = "hotpot_qa",
        max_observation_len: int = 1024,
        max_action_len: int = 1024,
        prompt_batch_size: int = 8,
        min_steps: int = 1,
        max_steps: int = 3,
        # max_actions: int,
        tool_ids: List[str] = ["google", "wikipedia"],
        num_trajectories_per_example: int = 2,
        # tools: List[Callable],
        tool_probs: Dict[str, float] = {"google": 0.5, "wikipedia": 0.5},
        # tokenizer: Callable,
        temperature: int = 0.7,
        do_sample: bool = True,
        num_beams: int = 1,
        num_samples: int = 5,
        num_few_shot_examples: Optional[int] = None,
    ) -> "PromptDataset":
        """Factory function."""

        # Set up language model.
        model = LLM(
            model="Qwen/Qwen2-72B-Instruct-AWQ",
            quantization="awq",
            max_model_len=16_384,
            max_num_seqs=32,
            gpu_memory_utilization=0.90,
        )

        max_prompt_len = model.llm_engine.model_config.max_model_len

        # Set up tokenizer.
        tokenizer = model.get_tokenizer()

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
            num_examples=num_few_shot_examples,
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
            },
            **{"max_length": max_prompt_len},
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
            max_steps=max_steps,
            max_prompt_len=max_prompt_len,
            max_generation_len=max_action_len,
            max_observation_len=max_observation_len,
        )

        # Build config for model generation
        sampling_params = SamplingParams(
            max_tokens=max_observation_len, skip_special_tokens=True
        )

        # Add actions column.
        dataset = dataset.add_column("actions", column=len(dataset) * [[]])

        # Build up trajectories
        for step in range(max_steps):

            ### Start GPU Phase ###
            dataset = cls.get_model_output(
                dataset,
                model=model,
                tokenizer=tokenizer,
                batch_size=prompt_batch_size,
                sampling_params=sampling_params,
            )
            ### End GPU Phase ###

            assert False

            # Validate model responses

            # Extract api calls from  model responses

            # Create and execute API calls

            # Clean and tokenize responses

            # Update prompt

            mock_calls = ["call1", "call2", "call3"]
            mock_resps = ["resp1", "resp2", "resp3"]

            # Add calls and responses to prompt.
            dataset = cls.update_prompt(
                cls,
                dataset,
                builder=prompt_builder,
                tokenizer=tokenizer,
                step=step,
                max_steps=max_steps,
                max_len_prompt_len=max_prompt_len,
                max_generation_len=max_action_len,
                max_observation_len=max_observation_len,
                calls=mock_calls,
                responses=mock_resps,
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
            if dataset_name == "hotpot_qa":
                dataset = load_dataset(
                    dataset_name,
                    "distractor",
                    split="train+validation",
                )

                if debug:
                    # CPU debugging.
                    dataset = dataset.select(range(4))

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
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must have pad token")

        def tokenize_fn(examples):
            encoding = tokenizer(
                examples[src_column],
                add_special_tokens=False,
                **kwargs,
            )

            result = {}

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
        max_steps: int,
        max_prompt_len: int,
        max_generation_len: int,
        max_observation_len: int,
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
            },
            **{"max_length": max_prompt_len},
        )

        self.validate_prompt_len(
            dataset,
            step=step,
            max_steps=max_steps,
            max_prompt_len=max_prompt_len,
            max_generation_len=max_generation_len,
            max_observation_len=max_observation_len,
        )

        return dataset

    @staticmethod
    def validate_prompt_len(
        dataset: datasets.Dataset,
        step: int,
        max_steps: int,
        max_prompt_len: int,
        max_generation_len: int,
        max_observation_len: int,
    ) -> None:
        max_valid_tokens = (
            dataset.to_polars().get_column("prompt_str").str.len_chars().max()
        )

        tokens_per_step = max_generation_len + max_observation_len

        # Final (nth) step will be "answer", and won't require model prompting
        remaining_steps = max_steps - step - 1

        if max_valid_tokens > (max_prompt_len - (remaining_steps * tokens_per_step)):
            raise ValueError(
                "Prompt is too long to allow generation of necessary subsequent prompts."
            )

    @staticmethod
    def get_model_output(
        dataset: datasets.Dataset,
        model: nn.Module,
        tokenizer: Callable,
        batch_size: int,
        sampling_params: GenerationConfig,
    ) -> datasets.Dataset:

        def model_forward(batch):
            # prompts = [
            #     TokensPrompt(prompt_token_ids=seq) for seq in batch["prompt_tokens"]
            # ]

            # output = model.generate(
            #     prompts,
            #     sampling_params=sampling_params,
            # )

            messages = [
                [
                    {"role": "system", "content": "You generate API calls..."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in batch["prompt_str"]
            ]

            output = model.chat(messages, sampling_params=sampling_params)

            token_output = [output[i].outputs[0].token_ids for i in range(len(output))]

            string_output = [output[i].outputs[0].text for i in range(len(output))]

            token_output = rearrange(
                np.array(token_output), "(b n) t -> b n t", b=batch_size
            ).tolist()

            string_output = rearrange(
                np.array(string_output), "(b n) -> b n", b=batch_size
            ).tolist()

            actions = batch["actions"]
            updated_actions = []

            for action_seq in actions:
                action_seq.append(token_output)
                updated_actions.append(action_seq)

            batch["actions"] = updated_actions
            batch["calls"] = string_output

            assert False

            return batch

        return dataset.map(
            model_forward,
            batched=True,
            batch_size=batch_size,
            cache_file_name=None,
        )

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
    PromptDataset.build_dataset()
