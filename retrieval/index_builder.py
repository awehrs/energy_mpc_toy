import json
from pathlib import Path
from typing import List

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class DenseIndexBuilder:
    """
    Builds dense FAISS index from preprocessed chunks using HuggingFace transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading HuggingFace model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_chunks(self, chunks: List[dict]) -> np.ndarray:
        """
        Encode chunk texts into dense embeddings using HuggingFace transformers.

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            embeddings: [n_chunks, embedding_dim] numpy array
        """
        texts = [chunk["text"] for chunk in chunks]
        all_embeddings = []

        print(f"Encoding {len(texts)} chunks...")
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)

                # Compute embeddings
                model_output = self.model(**encoded_input)
                
                # Apply mean pooling
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize if requested
                if self.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                all_embeddings.append(batch_embeddings.cpu())
                
                if (i // self.batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_texts)}/{len(texts)} chunks")

        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        print(f"Encoding complete: {embeddings.shape}")
        
        return embeddings

    def build_index(
        self,
        chunks: List[dict],
        index_type: str = "flat",
    ) -> faiss.Index:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")

        Returns:
            FAISS index
        """
        # Encode chunks
        embeddings = self.encode_chunks(chunks)

        # Build index
        print(f"Building {index_type} FAISS index...")

        if index_type == "flat":
            if self.normalize_embeddings:
                index = faiss.IndexFlatIP(
                    self.embedding_dim
                )  # Inner product for normalized vectors
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
        elif index_type == "ivf":
            # IVF index for large datasets
            nlist = min(4096, len(chunks) // 100)  # Number of clusters
            quantizer = (
                faiss.IndexFlatIP(self.embedding_dim)
                if self.normalize_embeddings
                else faiss.IndexFlatL2(self.embedding_dim)
            )
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.train(embeddings)
        elif index_type == "hnsw":
            # HNSW index for fast approximate search
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add embeddings to index
        print(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)

        print(f"Index built with {index.ntotal} vectors")
        return index

    def save_index_and_metadata(
        self,
        index: faiss.Index,
        chunks: List[dict],
        output_dir: Path,
        index_name: str = "chunk_index",
    ):
        """
        Save FAISS index and metadata.

        Args:
            index: FAISS index
            chunks: List of chunk dictionaries
            output_dir: Output directory
            index_name: Base name for index files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = output_dir / f"{index_name}.faiss"
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")

        # Save chunk ID mapping (for converting FAISS indices back to chunk IDs)
        chunk_id_mapping = [chunk["chunk_id"] for chunk in chunks]
        mapping_path = output_dir / f"{index_name}_chunk_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(chunk_id_mapping, f)
        print(f"Saved chunk mapping to {mapping_path}")

        # Save index metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "n_chunks": len(chunks),
            "normalize_embeddings": self.normalize_embeddings,
            "index_type": type(index).__name__,
        }
        metadata_path = output_dir / f"{index_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        return index_path, mapping_path, metadata_path


def build_chunk_index(
    chunks_file: Path,
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "flat",
    batch_size: int = 32,
    index_name: str = "chunk_index",
    device: str = "auto",
):
    """
    Main function to build dense index from preprocessed chunks.

    Args:
        chunks_file: Path to preprocessed chunks (.pt file)
        output_dir: Output directory for index files
        model_name: HuggingFace model name (e.g., sentence-transformers/all-MiniLM-L6-v2)
        index_type: Type of FAISS index
        batch_size: Batch size for encoding
        index_name: Base name for index files
        device: Device to use for encoding ("auto", "cuda", "cpu")
    """
    # Load chunks
    print(f"Loading chunks from {chunks_file}...")
    chunks = torch.load(chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    # Build index
    builder = DenseIndexBuilder(
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
    )

    index = builder.build_index(chunks, index_type=index_type)

    # Save index and metadata
    index_path, mapping_path, metadata_path = builder.save_index_and_metadata(
        index, chunks, output_dir, index_name
    )

    return index_path, mapping_path, metadata_path


if __name__ == "__main__":
    # Example usage
    chunks_file = Path("data/processed/hotpotqa_train/chunks.pt")
    output_dir = Path("data/indexes/hotpotqa_train")

    build_chunk_index(chunks_file, output_dir)
