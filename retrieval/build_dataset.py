#!/usr/bin/env python3
"""
Script to build HotpotQA dataset pipeline:
0. Download HotpotQA data (optional)
1. Preprocess HotpotQA data into chunks
2. Build dense index using SentenceTransformers
3. Create controlled retrieval dataset
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from retrieval.preprocess import preprocess_hotpotqa_dataset
from retrieval.index_builder import build_chunk_index
from retrieval.dataset import create_controlled_dataset
from retrieval.download import download_hotpotqa

logger = logging.getLogger(__name__)


def build_dataset_pipeline(cfg: DictConfig) -> Path:
    """Build complete HotpotQA pipeline."""

    if "dataset" not in cfg or cfg.dataset is None:
        raise ValueError("Please specify a dataset config (e.g., dataset=hotpotqa)")

    dataset_cfg = cfg.dataset

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    logger.info("Building HotpotQA pipeline with config: %s", dataset_cfg)

    # Determine if we're in debug mode based on training config name
    debug_mode = cfg.dataset.debug == True

    # Step 0: Check if raw data exists, suggest download if missing
    raw_data_path = Path("data/hotpotqa/hotpot_train_v1.1.json")
    processed_dir = Path("data/processed/hotpotqa_train")

    if not raw_data_path.exists():
        logger.warning("Raw data not found at %s", raw_data_path)
        logger.info("Downloading HotpotQA dataset...")

        # Download the dataset
        data_dir = Path("data/hotpotqa")
        download_hotpotqa(
            data_dir=data_dir, debug=debug_mode, debug_size=100 if debug_mode else None
        )

        # Verify the file now exists
        if not raw_data_path.exists():
            logger.error("Download failed: %s still not found", raw_data_path)
            return

    # Step 1: Preprocess raw data into chunks
    if not (processed_dir / "chunks.pt").exists():
        logger.info("Step 1: Preprocessing HotpotQA data...")
        chunks_file, examples_file = preprocess_hotpotqa_dataset(
            raw_data_path=raw_data_path,
            output_dir=processed_dir,
            pretrained_model_name=dataset_cfg.preprocessing.model_name,
            chunk_size=dataset_cfg.preprocessing.chunk_size,
            overlap_size=dataset_cfg.preprocessing.overlap_size,
            n_docs=dataset_cfg.n_docs,
        )
        logger.info("Preprocessing complete: %s, %s", chunks_file, examples_file)
    else:
        logger.info("Step 1: Chunks already exist, skipping preprocessing")
        chunks_file = processed_dir / "chunks.pt"
        examples_file = processed_dir / "examples.json"

    # Step 2: Build dense index
    index_dir = Path("data/indexes/hotpotqa_train")
    index_file = index_dir / f"{dataset_cfg.index_name}.faiss"

    if not index_file.exists():
        logger.info("Step 2: Building dense index...")
        index_path, mapping_path, metadata_path = build_chunk_index(
            chunks_file=chunks_file,
            output_dir=index_dir,
            model_name=dataset_cfg.index_building.model_name,
            index_type=dataset_cfg.index_building.index_type,
            batch_size=dataset_cfg.index_building.batch_size,
            index_name=dataset_cfg.index_name,
            device=dataset_cfg.index_building.device,
        )
        logger.info("Index building complete: %s", index_path)
    else:
        logger.info("Step 2: Index already exists, skipping index building")

    # Step 3: Create dataset (this just validates everything loads correctly)
    logger.info("Step 3: Creating controlled retrieval dataset...")
    dataset = create_controlled_dataset(
        chunks_file=processed_dir / "chunks.pt",
        examples_file=processed_dir / "examples.json",
        index_dir=index_dir,
        index_name=dataset_cfg.index_name,
        n_docs=dataset_cfg.n_docs,
        max_steps=dataset_cfg.max_steps,
        random_seed=dataset_cfg.random_seed,
    )

    logger.info("Dataset created successfully:")
    logger.info("  - %d examples", len(dataset))
    logger.info("  - %d docs per step", dataset.n_docs)
    logger.info("  - %d max steps", dataset.max_steps)
    logger.info("  - %d embedding dimension", dataset.index_dim)

    # Test loading one example
    logger.debug("Testing dataset loading...")
    example = dataset[0]
    logger.debug("Example shapes:")
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            logger.debug("  %s: %s", key, value.shape)
        else:
            logger.debug("  %s: %s - %s...", key, type(value), str(value)[:100])

    logger.info("Pipeline complete!")
    return dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """CLI entry point for building dataset."""
    build_dataset_pipeline(cfg)


if __name__ == "__main__":
    main()
