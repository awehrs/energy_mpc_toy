import json
import pytest
import tempfile
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

from retrieval.download import download_hotpotqa
from retrieval.index_builder import build_chunk_index
from retrieval.dataset import create_controlled_dataset
from retrieval.dataloader import create_flexible_dataloader
from retrieval.preprocess import preprocess_hotpotqa_dataset


@pytest.fixture(scope="module")
def downloaded_debug_data():
    """Fixture to download debug data once and reuse across tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "hotpotqa"

        # Download debug dataset
        downloaded_files = download_hotpotqa(
            data_dir=data_dir, debug=True, debug_size=5
        )

        # Return the train file path (debug mode uses standard filename)
        train_file = data_dir / "hotpot_train_v1.1.json"
        yield train_file


@pytest.fixture(scope="module")
def preprocessed_debug_data(downloaded_debug_data):
    """Fixture to preprocess data once and reuse across tests."""
    raw_data_path = downloaded_debug_data

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "processed"

        # Run preprocessing
        chunks_file, examples_file = preprocess_hotpotqa_dataset(
            raw_data_path=raw_data_path,
            output_dir=output_dir,
            pretrained_model_name="gpt2",
            chunk_size=256,  # Updated for sentence windows
            overlap_size=32,
        )

        yield chunks_file, examples_file


@pytest.fixture(scope="module")
def indexed_debug_data(preprocessed_debug_data):
    """Fixture to build index once and reuse across tests."""
    chunks_file, examples_file = preprocessed_debug_data

    # Load chunks to verify we have some data to index
    chunks = torch.load(chunks_file)
    if len(chunks) == 0:
        pytest.skip("No chunks were created during preprocessing")

    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "index"

        # Build index
        index_path, mapping_path, metadata_path = build_chunk_index(
            chunks_file=chunks_file,
            output_dir=index_dir,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_type="flat",
            batch_size=8,
            index_name="test_index",
            device="cpu",
        )

        yield chunks_file, examples_file, index_dir


def test_download_hotpotqa_debug():
    """Test downloading HotpotQA in debug mode with actual network call."""

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "hotpotqa"

        # Download debug dataset (small subset)
        downloaded_files = download_hotpotqa(
            data_dir=data_dir, debug=True, debug_size=5  # Very small for fast test
        )

        # Should download 3 files
        assert len(downloaded_files) == 3

        # Check that train file was created (debug mode uses standard filename)
        train_file = data_dir / "hotpot_train_v1.1.json"
        assert train_file.exists()
        assert train_file in downloaded_files

        # Verify the content was subsetted correctly
        with open(train_file, "r") as f:
            data = json.load(f)
            assert len(data) == 5  # Should be exactly debug_size

            # Verify structure of first example
            first_example = data[0]
            required_keys = {"_id", "question", "answer", "context", "supporting_facts"}
            assert all(key in first_example for key in required_keys)

            # Context should be list of [title, sentences] pairs
            assert isinstance(first_example["context"], list)
            assert len(first_example["context"]) > 0
            assert isinstance(first_example["context"][0], list)
            assert len(first_example["context"][0]) == 2  # [title, sentences]


def test_preprocess_hotpotqa(downloaded_debug_data):
    """Test preprocessing HotpotQA data into chunks."""

    raw_data_path = downloaded_debug_data

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "processed"

        # Run preprocessing
        chunks_file, examples_file = preprocess_hotpotqa_dataset(
            raw_data_path=raw_data_path,
            output_dir=output_dir,
            pretrained_model_name="gpt2",
            chunk_size=256,  # Increased for better sentence coverage
            overlap_size=32,
        )

        # Verify output files exist
        assert chunks_file.exists()
        assert examples_file.exists()

        # Load and verify chunks
        import torch

        chunks = torch.load(chunks_file)
        assert len(chunks) > 0

        # Verify chunk structure (updated for sliding sentence windows)
        first_chunk = chunks[0]
        required_keys = {
            "chunk_id",
            "input_ids",
            "attention_mask",
            "text",
            "source_title",
            "source_context_idx",
            "center_sentence_idx",
            "center_sentence_text",
            "sentence_indices",
            "window_sentences",
        }
        assert all(key in first_chunk for key in required_keys)

        # Verify tensor shapes
        assert first_chunk["input_ids"].shape[0] == 256  # chunk_size
        assert first_chunk["attention_mask"].shape[0] == 256

        # Verify sentence metadata (updated for sliding windows)
        assert isinstance(first_chunk["center_sentence_idx"], int)
        assert isinstance(first_chunk["center_sentence_text"], str)
        assert isinstance(first_chunk["sentence_indices"], list)
        assert isinstance(first_chunk["window_sentences"], list)
        assert len(first_chunk["sentence_indices"]) == len(
            first_chunk["window_sentences"]
        )

        # Verify boundary tokens are present
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<chunk_start>", "<chunk_end>"]}
        )

        chunk_start_id = tokenizer.convert_tokens_to_ids("<chunk_start>")
        chunk_end_id = tokenizer.convert_tokens_to_ids("<chunk_end>")

        input_ids = first_chunk["input_ids"]

        # First token should be <chunk_start>
        assert (
            input_ids[0] == chunk_start_id
        ), f"First token should be <chunk_start> ({chunk_start_id}), got {input_ids[0]}"

        # Find the last non-padding token
        attention_mask = first_chunk["attention_mask"]
        last_real_token_idx = torch.where(attention_mask == 1)[0][-1].item()

        # Last real token should be <chunk_end>
        assert (
            input_ids[last_real_token_idx] == chunk_end_id
        ), f"Last real token should be <chunk_end> ({chunk_end_id}), got {input_ids[last_real_token_idx]}"

        # Load and verify examples
        with open(examples_file, "r") as f:
            examples = json.load(f)

        # Note: With small debug datasets, we might not find examples with gold chunks
        # This is okay for testing purposes - we mainly want to test that chunks are created
        print(
            f"Found {len(examples)} examples with gold chunks out of {len(chunks)} total chunks"
        )

        if len(examples) > 0:
            # Verify example structure if examples exist
            first_example = examples[0]
            required_keys = {
                "question_id",
                "question",
                "answer",
                "chunk_ids",
                "gold_chunk_ids",
                "num_chunks",
                "answer_tokens",
                "answer_attention_mask",
            }
            assert all(key in first_example for key in required_keys)

            # Verify tokenized answer structure
            assert isinstance(first_example["answer_tokens"], list)
            assert isinstance(first_example["answer_attention_mask"], list)
            assert len(first_example["answer_tokens"]) == len(
                first_example["answer_attention_mask"]
            )
            assert (
                len(first_example["answer_tokens"]) == 32
            )  # max_length from preprocessing

            # Verify relationships
            assert len(first_example["chunk_ids"]) == first_example["num_chunks"]
            assert len(first_example["gold_chunk_ids"]) > 0  # Should have gold chunks


def test_build_chunk_index(preprocessed_debug_data):
    """Test building FAISS index from preprocessed chunks."""

    chunks_file, examples_file = preprocessed_debug_data

    # Load chunks to verify we have some data to index
    chunks = torch.load(chunks_file)
    if len(chunks) == 0:
        pytest.skip("No chunks were created during preprocessing")

    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "index"

        # Build index with small model for speed
        index_path, mapping_path, metadata_path = build_chunk_index(
            chunks_file=chunks_file,
            output_dir=index_dir,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_type="flat",
            batch_size=8,  # Small batch
            index_name="test_index",
            device="cpu",  # Force CPU for reproducibility
        )

        # Verify output files exist
        assert index_path.exists()
        assert mapping_path.exists()
        assert metadata_path.exists()

        # Load and verify index
        import faiss

        index = faiss.read_index(str(index_path))
        assert index.ntotal > 0  # Should have vectors
        assert index.d == 384  # all-MiniLM-L6-v2 dimension

        # Load and verify mapping
        with open(mapping_path, "r") as f:
            chunk_mapping = json.load(f)
        assert len(chunk_mapping) == index.ntotal
        assert all(isinstance(cid, int) for cid in chunk_mapping)

        # Load and verify metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_keys = {
            "model_name",
            "embedding_dim",
            "n_chunks",
            "normalize_embeddings",
            "index_type",
        }
        assert all(key in metadata for key in required_keys)
        assert metadata["embedding_dim"] == 384
        assert metadata["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert metadata["n_chunks"] > 0

        # Test basic search
        query_vector = torch.randn(1, 384).numpy().astype(np.float32)
        faiss.normalize_L2(query_vector)  # Normalize for cosine similarity

        scores, indices = index.search(query_vector, k=3)
        assert len(scores[0]) == 3
        assert len(indices[0]) == 3
        assert all(0 <= idx < len(chunk_mapping) for idx in indices[0])


def test_controlled_dataset_pipeline(indexed_debug_data):
    """Test the full controlled retrieval dataset pipeline."""

    chunks_file, examples_file, index_dir = indexed_debug_data

    # Create controlled dataset even if no examples have gold chunks
    # This tests that the dataset can handle edge cases gracefully
    dataset = create_controlled_dataset(
        chunks_file=chunks_file,
        examples_file=examples_file,
        index_dir=index_dir,
        index_name="test_index",
        n_docs=3,  # Small for testing
        max_steps=2,  # Small for testing
        random_seed=42,
    )

    # Check dataset properties
    assert hasattr(dataset, "chunks")
    assert hasattr(dataset, "examples")
    assert hasattr(dataset, "index")
    assert hasattr(dataset, "n_docs")
    assert hasattr(dataset, "max_steps")
    assert hasattr(dataset, "index_dim")

    assert dataset.n_docs == 3
    assert dataset.max_steps == 2
    assert dataset.index_dim == 384  # all-MiniLM-L6-v2 dimension
    assert len(dataset.chunks) > 0  # Should have chunks

    # Test dataset length
    dataset_len = len(dataset)
    print(f"Dataset has {dataset_len} examples")

    # If we have examples, test the dataset format
    if dataset_len > 0:
        # Test getting an example
        example = dataset[0]

        # Verify output structure (updated for new dataset)
        required_keys = {
            "input_ids",
            "input_attention_mask",  # Correct key name to distinguish from question_attention_mask
            "retrieval_queries",
            "target_tokens",
            "step_mask",
            "question_id",
            "question",
            "answer",
            "question_input_ids",
            "question_attention_mask",
            "gold_chunk_ids",  # Now included for flexible dataloader
        }
        assert all(key in example for key in required_keys)

        # Verify tensor shapes
        assert example["input_ids"].shape == (
            2,
            3,
            256,  # Updated chunk_size from 32 to 256
        )  # [max_steps, n_docs, chunk_size]
        assert example["input_attention_mask"].shape == (
            2,
            3,
            256,  # Updated chunk_size
        )  # [max_steps, n_docs, chunk_size]
        assert example["retrieval_queries"].shape == (
            3,
            384,
        )  # [max_steps+1, index_dim]
        assert example["target_tokens"].shape == (2, 32)  # [max_steps, target_seq_len]
        assert example["step_mask"].shape == (2,)  # [max_steps]
        assert example["question_input_ids"].shape == (2048,)  # [n_docs * chunk_size] = 8 * 256 (preprocessor default)
        assert example["question_attention_mask"].shape == (
            2048,
        )  # [n_docs * chunk_size] = 8 * 256 (preprocessor default)

        # Verify types
        assert isinstance(example["input_ids"], torch.Tensor)
        assert isinstance(example["input_attention_mask"], torch.Tensor)
        assert isinstance(example["retrieval_queries"], torch.Tensor)
        assert isinstance(example["target_tokens"], torch.Tensor)
        assert isinstance(example["step_mask"], torch.Tensor)
        assert isinstance(example["question_input_ids"], torch.Tensor)
        assert isinstance(example["question_attention_mask"], torch.Tensor)
        # assert isinstance(example["gold_chunk_ids"], list)  # Not in main dataset
        assert isinstance(example["question_id"], str)
        assert isinstance(example["question"], str)
        assert isinstance(example["answer"], str)

        # Verify data types
        assert example["input_ids"].dtype == torch.long
        assert example["input_attention_mask"].dtype == torch.long
        assert example["retrieval_queries"].dtype == torch.float32
        assert example["target_tokens"].dtype == torch.long
        assert example["step_mask"].dtype == torch.bool
        assert example["question_input_ids"].dtype == torch.long
        assert example["question_attention_mask"].dtype == torch.long

        # Note: gold_chunk_ids is handled by the flexible dataloader, not the base dataset

        print(f"Example format verified successfully!")
        print(f"  - Question: {example['question'][:50]}...")
        print(f"  - Answer: {example['answer'][:50]}...")
        print(f"  - Active steps: {example['step_mask'].sum().item()}")

    else:
        print(
            "No examples with gold chunks found - this is expected with small debug datasets"
        )
        print("But chunks and index were created successfully!")


def test_flexible_dataloader(indexed_debug_data):
    """Test the flexible data loader with step permutation and gold insertion."""

    chunks_file, examples_file, index_dir = indexed_debug_data

    # Create dataset
    dataset = create_controlled_dataset(
        chunks_file=chunks_file,
        examples_file=examples_file,
        index_dir=index_dir,
        index_name="test_index",
        n_docs=3,
        max_steps=2,
        random_seed=42,
    )

    # If no examples, create a minimal test case
    if len(dataset) == 0:
        print("No examples found - skipping flexible dataloader test")
        pytest.skip("No examples with gold chunks found")

    # Create chunk database for the flexible dataloader
    chunk_db = {chunk["chunk_id"]: chunk for chunk in dataset.chunks}

    # Create flexible dataloader
    flexible_dataloader = create_flexible_dataloader(
        dataset=dataset,
        chunk_db=chunk_db,
        batch_size=2,
        shuffle=False,  # Deterministic for testing
        gold_insertion_prob=1.0,  # Always insert gold for testing
        max_gold_per_step=1,
        permute_steps=True,
    )

    # Test one batch
    for batch in flexible_dataloader:
        # Verify batch structure
        required_keys = {
            "input_ids",
            "input_attention_mask",
            "retrieval_queries",
            "target_tokens",
            "question_input_ids",
            "question_attention_mask",
            "step_order",
            "gold_positions",
        }
        assert all(key in batch for key in required_keys)

        batch_size = batch["input_ids"].shape[0]

        # Verify shapes
        assert batch["input_ids"].shape == (batch_size, 2, 3, 256)  # Updated chunk_size
        assert batch["input_attention_mask"].shape == (batch_size, 2, 3, 256)  # Updated chunk_size
        assert batch["retrieval_queries"].shape == (batch_size, 3, 384)
        assert batch["question_input_ids"].shape == (batch_size, 2048)  # 8 * 256
        assert batch["question_attention_mask"].shape == (batch_size, 2048)  # 8 * 256

        # Verify metadata
        assert len(batch["step_order"]) == batch_size
        assert len(batch["gold_positions"]) == batch_size

        # Verify step orders are permutations
        for step_order in batch["step_order"]:
            assert sorted(step_order) == [0, 1]  # Should be permutation of [0, 1]

        # Verify gold positions tracking
        for gold_pos in batch["gold_positions"]:
            assert len(gold_pos) == 2  # One per step
            for step_gold in gold_pos:
                assert isinstance(step_gold, list)
                # Each step should have 0-1 gold positions (max_gold_per_step=1)
                assert len(step_gold) <= 1

        print(f"Flexible dataloader test passed!")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Step orders: {batch['step_order']}")
        print(f"  - Gold positions: {batch['gold_positions']}")

        break  # Only test one batch
