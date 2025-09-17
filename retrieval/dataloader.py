"""
Flexible data loader with step permutation and dynamic gold insertion.
"""

import random
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader


class FlexibleDataCollator:
    """
    Custom collate function that applies step permutation and dynamic gold insertion.
    """

    def __init__(
        self,
        chunk_db: Dict[int, Dict[str, Any]],
        gold_insertion_prob: float = 0.8,
        max_gold_per_step: int = 2,
        permute_steps: bool = True,
        seed: int = None,
    ):
        self.chunk_db = chunk_db
        self.gold_insertion_prob = gold_insertion_prob
        self.max_gold_per_step = max_gold_per_step
        self.permute_steps = permute_steps
        self.rng = np.random.RandomState(seed)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Apply step permutation and gold insertion to a batch of examples.

        Args:
            batch: List of examples from dataset

        Returns:
            Batch dict with permuted and gold-augmented sequences
        """
        batch_size = len(batch)
        processed_batch = []

        for example in batch:
            processed_example = self._process_single_example(example)
            processed_batch.append(processed_example)

        # Stack all examples into batch tensors
        return self._stack_batch(processed_batch)

    def _process_single_example(self, example: Dict) -> Dict:
        """Process a single example with permutation and gold insertion."""

        # Extract base data
        input_ids = example["input_ids"]  # [n_steps, n_docs, seq_len]
        attention_mask = example["input_attention_mask"]  # [n_steps, n_docs, seq_len]
        gold_chunk_ids = example["gold_chunk_ids"]
        n_steps = input_ids.shape[0]

        # Generate step permutation
        if self.permute_steps:
            step_order = list(range(n_steps))
            self.rng.shuffle(step_order)
        else:
            step_order = list(range(n_steps))

        # Apply permutation to input data
        permuted_input_ids = input_ids[step_order]
        permuted_attention_mask = attention_mask[step_order]

        # For each step, decide whether to insert gold and how much
        modified_input_ids = []
        modified_attention_mask = []
        gold_positions = []  # Track where gold was inserted

        for step_idx in range(n_steps):
            step_input_ids = permuted_input_ids[step_idx]  # [n_docs, seq_len]
            step_attention_mask = permuted_attention_mask[step_idx]

            # Decide if this step gets gold chunks
            if gold_chunk_ids and self.rng.random() < self.gold_insertion_prob:
                # How many gold chunks to insert
                n_gold_to_insert = self.rng.randint(
                    1,
                    min(
                        self.max_gold_per_step + 1,
                        len(gold_chunk_ids) + 1,
                        step_input_ids.shape[0] + 1,
                    ),
                )

                # Select which gold chunks
                selected_gold_ids = self.rng.choice(
                    gold_chunk_ids, size=n_gold_to_insert, replace=False
                )

                # Select positions to replace
                replacement_positions = self.rng.choice(
                    step_input_ids.shape[0], size=n_gold_to_insert, replace=False
                )

                # Replace with gold chunks
                modified_step_input = step_input_ids.clone()
                modified_step_attention = step_attention_mask.clone()

                for pos, gold_id in zip(replacement_positions, selected_gold_ids):
                    gold_chunk = self.chunk_db[
                        int(gold_id)
                    ]  # Convert numpy int to Python int
                    modified_step_input[pos] = torch.tensor(gold_chunk["input_ids"])
                    modified_step_attention[pos] = torch.tensor(
                        gold_chunk["attention_mask"]
                    )

                # Track gold positions for this step
                step_gold_positions = [int(pos) for pos in replacement_positions]
            else:
                modified_step_input = step_input_ids
                modified_step_attention = step_attention_mask
                step_gold_positions = []

            modified_input_ids.append(modified_step_input)
            modified_attention_mask.append(modified_step_attention)
            gold_positions.append(step_gold_positions)

        # Stack back into tensor format
        final_input_ids = torch.stack(modified_input_ids)
        final_attention_mask = torch.stack(modified_attention_mask)

        # Apply same permutation to other sequence data (skip last retrieval query since it's meaningless)
        retrieval_queries_steps = example["retrieval_queries"][
            :-1
        ]  # [n_steps, index_dim]
        permuted_retrieval_queries_steps = retrieval_queries_steps[step_order]
        # Add back the final meaningless query
        permuted_retrieval_queries = torch.cat(
            [permuted_retrieval_queries_steps, example["retrieval_queries"][-1:]], dim=0
        )
        permuted_target_tokens = (
            example["target_tokens"][step_order] if "target_tokens" in example else None
        )

        return {
            "input_ids": final_input_ids,
            "input_attention_mask": final_attention_mask,
            "retrieval_queries": permuted_retrieval_queries,
            "target_tokens": permuted_target_tokens,
            "question_input_ids": example["question_input_ids"],
            "question_attention_mask": example["question_attention_mask"],
            "step_order": step_order,
            "gold_positions": gold_positions,
        }

    def _stack_batch(self, processed_batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Stack processed examples into batch format."""
        batch_dict = {}

        # Handle tensor fields
        tensor_fields = [
            "input_ids",
            "input_attention_mask",
            "retrieval_queries",
            "question_input_ids",
            "question_attention_mask",
        ]

        for field in tensor_fields:
            if processed_batch[0][field] is not None:
                batch_dict[field] = torch.stack([ex[field] for ex in processed_batch])

        # Handle target_tokens which might be None
        if processed_batch[0]["target_tokens"] is not None:
            batch_dict["target_tokens"] = torch.stack(
                [ex["target_tokens"] for ex in processed_batch]
            )

        # Handle metadata (lists)
        batch_dict["step_order"] = [ex["step_order"] for ex in processed_batch]
        batch_dict["gold_positions"] = [ex["gold_positions"] for ex in processed_batch]

        return batch_dict


def create_flexible_dataloader(
    dataset,
    chunk_db: Dict[int, Dict[str, Any]],
    batch_size: int = 4,
    shuffle: bool = True,
    gold_insertion_prob: float = 0.8,
    max_gold_per_step: int = 2,
    permute_steps: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with flexible gold insertion and step permutation.

    Args:
        dataset: Base dataset (should return examples without gold insertion)
        chunk_db: Dictionary mapping chunk_id -> chunk data
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        gold_insertion_prob: Probability of inserting gold in each step
        max_gold_per_step: Maximum gold chunks per step
        permute_steps: Whether to permute step order
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader with custom collate function
    """
    collate_fn = FlexibleDataCollator(
        chunk_db=chunk_db,
        gold_insertion_prob=gold_insertion_prob,
        max_gold_per_step=max_gold_per_step,
        permute_steps=permute_steps,
    )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs
    )
