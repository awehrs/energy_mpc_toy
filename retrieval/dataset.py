import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import Dataset


class ControlledHotpotQADataset(Dataset):
    """
    Controlled retrieval dataset for HotpotQA experiments.

    Uses KNN retrieval with random query vectors against a real dense index,
    with gold passages force-inserted in the final step.
    """

    def __init__(
        self,
        chunks_file: Path,
        examples_file: Path,
        index_file: Path,
        chunk_mapping_file: Path,
        n_docs: int = 8,
        max_steps: int = 8,
        index_dim: int = 384,  # Default for all-MiniLM-L6-v2
        random_seed: int = 42,
    ):
        self.n_docs = n_docs
        self.max_steps = max_steps
        self.index_dim = index_dim
        self.rng = np.random.RandomState(random_seed)

        # Load preprocessed data
        print(f"Loading chunks from {chunks_file}...")
        self.chunks = torch.load(chunks_file)

        print(f"Loading examples from {examples_file}...")
        with open(examples_file, "r") as f:
            self.examples = json.load(f)

        print(f"Loading chunk mapping from {chunk_mapping_file}...")
        with open(chunk_mapping_file, "r") as f:
            self.index_to_chunk_id = json.load(f)

        print(f"Loading FAISS index from {index_file}...")
        self.index = faiss.read_index(str(index_file))

        print(f"Loaded {len(self.chunks)} chunks, {len(self.examples)} examples")
        print(f"Index contains {self.index.ntotal} vectors")

        # Build mapping from chunk ID to chunk data
        self.chunk_id_to_data = {chunk["chunk_id"]: chunk for chunk in self.chunks}

        # Verify index dimension matches
        if self.index.d != self.index_dim:
            print(
                f"Warning: Index dimension ({self.index.d}) != specified index_dim ({self.index_dim})"
            )
            self.index_dim = self.index.d

    def _get_knn_chunks(self, query_vector: np.ndarray, k: int) -> List[int]:
        """Get k nearest neighbor chunk IDs for a random query vector."""
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        # Normalize query vector to match index (if index uses cosine similarity)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)
        return [self.index_to_chunk_id[idx] for idx in indices[0]]

    def _get_chunks_data(
        self, chunk_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input_ids and attention_mask tensors for given chunk IDs."""
        input_ids_list = []
        attention_mask_list = []

        for chunk_id in chunk_ids:
            chunk_data = self.chunk_id_to_data[chunk_id]
            input_ids_list.append(chunk_data["input_ids"])
            attention_mask_list.append(chunk_data["attention_mask"])

        # Stack into tensors: [n_docs, chunk_size]
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return input_ids, attention_mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one training example with controlled retrieval.

        Returns:
            Dictionary containing:
            - input_ids: [max_steps, n_docs, chunk_size] - retrieved documents per step
            - attention_mask: [max_steps, n_docs, chunk_size] - attention masks
            - question_input_ids: [n_docs * chunk_size] - tokenized question
            - question_attention_mask: [n_docs * chunk_size] - question attention mask
            - retrieval_queries: [max_steps+1, index_dim] - queries (including final meaningless one)
            - target_tokens: [max_steps, target_seq_len] - target sequences
            - step_mask: [max_steps] - which steps are valid
            - question_id: string - question identifier
            - question: string - question text
            - answer: string - answer text
        """
        example = self.examples[idx]

        # Generate random retrieval queries for all steps + 1 final meaningless query
        retrieval_queries = self.rng.randn(self.max_steps + 1, self.index_dim).astype(
            np.float32
        )

        # Get tokenized question (already padded to n_docs * chunk_size)
        question_tokens = torch.tensor(example["question_tokens"], dtype=torch.long)
        question_attention_mask = torch.tensor(
            example["question_attention_mask"], dtype=torch.long
        )

        # Collect retrieved chunks for each step
        all_input_ids = []
        all_attention_masks = []
        step_mask = torch.zeros(self.max_steps, dtype=torch.bool)

        for step in range(self.max_steps):

            # Use query[step] for retrieving docs after step 'step'
            query_vector = retrieval_queries[step]

            # Get KNN chunks using random query vector
            retrieved_chunk_ids = self._get_knn_chunks(query_vector, self.n_docs)

            # Get chunk data
            input_ids, attention_mask = self._get_chunks_data(retrieved_chunk_ids)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            step_mask[step] = True

        # Stack into final tensors
        input_ids = torch.stack(all_input_ids)  # [max_steps, n_docs, chunk_size]
        attention_mask = torch.stack(
            all_attention_masks
        )  # [max_steps, n_docs, chunk_size]
        retrieval_queries_tensor = torch.from_numpy(
            retrieval_queries
        )  # [max_steps+1, index_dim]

        # Use pre-tokenized answer tokens, repeated for each step
        answer_tokens = torch.tensor(example["answer_tokens"], dtype=torch.long)
        answer_attention_mask = torch.tensor(
            example["answer_attention_mask"], dtype=torch.long
        )
        target_tokens = answer_tokens.unsqueeze(0).repeat(
            self.max_steps, 1
        )  # [max_steps, seq_len]
        target_attention_mask = answer_attention_mask.unsqueeze(0).repeat(
            self.max_steps, 1
        )  # [max_steps, seq_len]

        return {
            "input_ids": input_ids,
            "input_attention_mask": attention_mask,
            "question_input_ids": question_tokens,
            "question_attention_mask": question_attention_mask,
            "retrieval_queries": retrieval_queries_tensor,
            "target_tokens": target_tokens,
            "target_attention_mask": target_attention_mask,
            "step_mask": step_mask,
            "question_id": example["question_id"],
            "question": example["question"],
            "answer": example["answer"],
            "gold_chunk_ids": example["gold_chunk_ids"],  # For dataloader
        }


class IndexedDatasetLoader:
    """
    Helper class to load indexed dataset components.
    """

    @staticmethod
    def load_metadata(metadata_file: Path) -> Dict:
        """Load index metadata."""
        with open(metadata_file, "r") as f:
            return json.load(f)

    @staticmethod
    def create_dataset(
        chunks_file: Path,
        examples_file: Path,
        index_dir: Path,
        index_name: str = "chunk_index",
        n_docs: int = 8,
        max_steps: int = 8,
        random_seed: int = 42,
    ) -> ControlledHotpotQADataset:
        """
        Create dataset from index directory.

        Args:
            chunks_file: Path to chunks.pt file
            examples_file: Path to examples.json file
            index_dir: Directory containing index files
            index_name: Base name for index files
            n_docs: Number of documents per retrieval step
            max_steps: Maximum number of retrieval steps
            random_seed: Random seed for reproducibility

        Returns:
            ControlledHotpotQADataset instance
        """
        # Construct file paths
        index_file = index_dir / f"{index_name}.faiss"
        chunk_mapping_file = index_dir / f"{index_name}_chunk_mapping.json"
        metadata_file = index_dir / f"{index_name}_metadata.json"

        # Load metadata to get index dimension
        metadata = IndexedDatasetLoader.load_metadata(metadata_file)
        index_dim = metadata["embedding_dim"]

        return ControlledHotpotQADataset(
            chunks_file=chunks_file,
            examples_file=examples_file,
            index_file=index_file,
            chunk_mapping_file=chunk_mapping_file,
            n_docs=n_docs,
            max_steps=max_steps,
            index_dim=index_dim,
            random_seed=random_seed,
        )


def create_controlled_dataset(
    chunks_file: Path,
    examples_file: Path,
    index_dir: Path,
    index_name: str = "chunk_index",
    n_docs: int = 8,
    max_steps: int = 8,
    random_seed: int = 42,
) -> ControlledHotpotQADataset:
    """
    Create a controlled HotpotQA dataset for retrieval experiments.

    Args:
        chunks_file: Path to preprocessed chunks (.pt file)
        examples_file: Path to preprocessed examples (.json file)
        index_dir: Directory containing FAISS index files
        index_name: Base name for index files
        n_docs: Number of documents to retrieve per step
        max_steps: Maximum number of retrieval steps
        random_seed: Random seed for reproducibility

    Returns:
        ControlledHotpotQADataset instance
    """
    return IndexedDatasetLoader.create_dataset(
        chunks_file=chunks_file,
        examples_file=examples_file,
        index_dir=index_dir,
        index_name=index_name,
        n_docs=n_docs,
        max_steps=max_steps,
        random_seed=random_seed,
    )
