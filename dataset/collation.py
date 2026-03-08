import os
from typing import Dict, List

import torch


class TrajectoryCollator:
    """Collates variable-length trajectories into packed flash-attention format.

    Produces:
    1. Packed token sequences (flash-attention cu_seq_lens format) for sensor encoding.
    2. Trajectory boundary metadata for post-encoding padding before memory processing.

    Each sample is expected to yield:
        {"action_tokens": [[...], [...], ...], "precept_tokens": [[...], [...], ...]}
    where len(action_tokens) == len(precept_tokens) == trajectory_length.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        traj_lens = []
        all_precept_seqs = []
        all_action_seqs = []

        for sample in features:
            action_steps = sample["action_tokens"]
            precept_steps = sample["precept_tokens"]
            assert len(action_steps) == len(precept_steps)
            traj_lens.append(len(action_steps))
            all_precept_seqs.extend(precept_steps)
            all_action_seqs.extend(action_steps)

        batch_size = len(features)
        max_traj_len = max(traj_lens)

        precept_ids, precept_cu_seq_lens, precept_position_ids, precept_max_seq_len = (
            self._pack_sequences(all_precept_seqs)
        )
        action_ids, action_cu_seq_lens, action_position_ids, action_max_seq_len = (
            self._pack_sequences(all_action_seqs)
        )

        traj_lens_t = torch.tensor(traj_lens, dtype=torch.long)
        cu_traj_lens = torch.zeros(batch_size + 1, dtype=torch.int32)
        cu_traj_lens[1:] = torch.cumsum(traj_lens_t.to(torch.int32), dim=0)
        traj_mask = torch.arange(max_traj_len).unsqueeze(0) < traj_lens_t.unsqueeze(1)

        return {
            "precept_ids": precept_ids,
            "precept_position_ids": precept_position_ids,
            "precept_cu_seq_lens": precept_cu_seq_lens,
            "precept_max_seq_len": precept_max_seq_len,
            "action_ids": action_ids,
            "action_position_ids": action_position_ids,
            "action_cu_seq_lens": action_cu_seq_lens,
            "action_max_seq_len": action_max_seq_len,
            "traj_lens": traj_lens_t,
            "max_traj_len": max_traj_len,
            "cu_traj_lens": cu_traj_lens,
            "traj_mask": traj_mask,
        }

    @staticmethod
    def _pack_sequences(sequences: List[List[int]]):
        """Flatten variable-length token sequences into flash-attention packed format.

        Returns:
            ids:           [total_tokens]    int64
            cu_seq_lens:   [num_seqs + 1]    int32
            position_ids:  [total_tokens]    int64
            max_seq_len:   int
        """
        all_ids, all_position_ids, offsets = [], [], [0]
        max_seq_len = 0
        for seq in sequences:
            n = len(seq)
            max_seq_len = max(max_seq_len, n)
            all_ids.extend(seq)
            all_position_ids.extend(range(n))
            offsets.append(offsets[-1] + n)
        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(all_position_ids, dtype=torch.long),
            max_seq_len,
        )


class BucketBatchSampler:
    """Groups trajectories by step-count into buckets, greedily packs by total token count.

    Ensures consistent GPU memory/compute across batches while minimising
    trajectory-level padding waste before the sequence model.

    Compatible with torch.utils.data.Subset and multi-GPU via WORLD_SIZE/RANK env vars.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_tokens: int,
        bucket_width: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.max_tokens = max_tokens
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_replicas = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.epoch = 0

        if isinstance(dataset, torch.utils.data.Subset):
            actual_data = dataset.dataset.data
            subset_indices = list(dataset.indices)
            all_traj_lengths = self._get_traj_lengths(actual_data)
            all_token_counts = self._get_token_counts(actual_data)
            self.traj_lengths = [all_traj_lengths[i] for i in subset_indices]
            self.token_counts = [all_token_counts[i] for i in subset_indices]
        else:
            self.traj_lengths = self._get_traj_lengths(dataset.data)
            self.token_counts = self._get_token_counts(dataset.data)

        self.buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(self.traj_lengths):
            bucket_id = length // self.bucket_width
            self.buckets.setdefault(bucket_id, []).append(idx)

    @staticmethod
    def _get_traj_lengths(data) -> List[int]:
        if hasattr(data, "column_names"):
            if "num_steps" in data.column_names:
                return list(data["num_steps"])
            return [len(seq) for seq in data["action_tokens"]]
        if isinstance(data, dict):
            if "num_steps" in data:
                return list(data["num_steps"])
            return [len(seq) for seq in data["action_tokens"]]
        raise ValueError("Cannot determine trajectory lengths from dataset")

    @staticmethod
    def _get_token_counts(data) -> List[int]:
        cols = data.column_names if hasattr(data, "column_names") else list(data.keys())
        if "action_tokens_length" in cols and "precept_tokens_length" in cols:
            return [
                sum(a) + sum(p)
                for a, p in zip(data["action_tokens_length"], data["precept_tokens_length"])
            ]
        if "action_tokens" in cols and "precept_tokens" in cols:
            return [
                sum(len(s) for s in a) + sum(len(s) for s in p)
                for a, p in zip(data["action_tokens"], data["precept_tokens"])
            ]
        raise ValueError("Cannot determine token counts from dataset")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            current = 0
            for idx in indices:
                tc = self.token_counts[idx]
                if current + tc > self.max_tokens and current > 0:
                    total += 1
                    current = 0
                current += tc
            if current > 0:
                total += 1
        return total // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_batches = []

        for bucket_id in sorted(self.buckets.keys()):
            indices = self.buckets[bucket_id].copy()
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            current_batch, current_tokens = [], 0
            for idx in indices:
                tc = self.token_counts[idx]
                if current_tokens + tc > self.max_tokens and current_batch:
                    all_batches.append(current_batch)
                    current_batch, current_tokens = [], 0
                current_batch.append(idx)
                current_tokens += tc
            if current_batch and not self.drop_last:
                all_batches.append(current_batch)

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        batches_per_replica = len(all_batches) // self.num_replicas
        start = self.rank * batches_per_replica
        end = start + batches_per_replica if self.rank < self.num_replicas - 1 else len(all_batches)
        yield from all_batches[start:end]
