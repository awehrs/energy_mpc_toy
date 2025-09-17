import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class HotpotQAPreprocessor:
    """
    Preprocesses HotpotQA data for controlled retrieval experiments.

    Creates fixed-size chunks (256 tokens) with overlap (32 tokens) from passages,
    tokenizes with pretrained LM tokenizer, and prepares gold chunk mappings.
    """

    def __init__(
        self,
        pretrained_model_name: str = "gpt2",
        chunk_size: int = 256,
        overlap_size: int = 32,
        min_chunk_length: int = 64,
        n_docs: int = 8,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens for chunk boundaries
        special_tokens = {"additional_special_tokens": ["<chunk_start>", "<chunk_end>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.chunk_start_token = "<chunk_start>"
        self.chunk_end_token = "<chunk_end>"

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_length = min_chunk_length
        self.n_docs = n_docs
        # Stride based on content size (excluding boundary tokens)
        self.content_size = chunk_size - 2  # Reserve 2 tokens for boundaries
        self.stride = self.content_size - overlap_size  # stride between content starts

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s\.\,\?\!\;\:\-\(\)]", "", text)
        return text

    def extract_passages_from_context(
        self, context: List[List[str]]
    ) -> List[Dict[str, str]]:
        """
        Extract individual passages from HotpotQA context format.

        Args:
            context: List of [title, [sentence1, sentence2, ...]] pairs

        Returns:
            List of passage dictionaries with metadata
        """
        passages = []

        for ctx_idx, (title, sentences) in enumerate(context):
            # Combine all sentences for this document
            full_text = " ".join(sentences)
            cleaned_text = self.clean_text(full_text)

            # Skip very short passages
            if (
                len(cleaned_text.split()) < self.min_chunk_length // 4
            ):  # rough word count
                continue

            passages.append(
                {
                    "text": cleaned_text,
                    "title": title,
                    "context_idx": ctx_idx,
                    "original_sentences": sentences,
                }
            )

        return passages

    def create_sentence_windows(
        self, original_sentences: List[str], window_size: int = 3
    ) -> List[Dict]:
        """
        Create sliding windows of sentences for chunking.
        Each window contains window_size sentences with the center sentence as the focus.

        Args:
            original_sentences: List of original sentences from the passage
            window_size: Number of sentences per window (should be odd, default 3)

        Returns:
            List of window dicts with sentence content and metadata
        """
        windows = []

        if not original_sentences:
            return windows

        # Ensure window_size is odd for clear center
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2

        # Create sliding windows over sentences
        for center_idx in range(len(original_sentences)):

            # Determine window boundaries
            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(original_sentences), center_idx + half_window + 1)

            # Extract sentences for this window
            window_sentences = []
            sentence_indices = []

            # Add padding if needed at the beginning
            while len(window_sentences) < half_window - center_idx:
                window_sentences.append("")  # Empty padding
                sentence_indices.append(-1)  # Invalid index for padding

            # Add actual sentences
            for idx in range(start_idx, end_idx):
                sentence_clean = original_sentences[idx].strip()
                window_sentences.append(sentence_clean)
                sentence_indices.append(idx)

            # Add padding if needed at the end
            while len(window_sentences) < window_size:
                window_sentences.append("")  # Empty padding
                sentence_indices.append(-1)  # Invalid index for padding

            # Join sentences for this window
            window_text = " ".join(sent for sent in window_sentences if sent.strip())

            windows.append(
                {
                    "center_sentence_idx": center_idx,
                    "sentence_indices": sentence_indices,
                    "sentences": window_sentences,
                    "text": window_text,
                    "center_sentence_text": original_sentences[center_idx].strip(),
                }
            )

        return windows

    def create_chunks_from_sentences(self, source_info: Dict) -> List[Dict]:
        """
        Create chunks using sliding sentence windows.

        Args:
            source_info: Metadata about the source (title, context_idx, original_sentences, etc.)

        Returns:
            List of chunk dictionaries with tokenized content and clear sentence metadata
        """
        # Get original sentences for this passage
        original_sentences = source_info.get("original_sentences", [])

        if not original_sentences:
            return []

        # Create sentence windows
        windows = self.create_sentence_windows(original_sentences, window_size=3)

        chunks = []
        chunk_start_id = self.tokenizer.convert_tokens_to_ids(self.chunk_start_token)
        chunk_end_id = self.tokenizer.convert_tokens_to_ids(self.chunk_end_token)

        for chunk_id, window in enumerate(windows):
            # Skip empty windows
            if not window["text"].strip():
                continue

            # Tokenize the window text
            window_tokens = self.tokenizer(
                window["text"], add_special_tokens=False, return_tensors="pt"
            )
            content_tokens = window_tokens["input_ids"].squeeze(0)

            # Truncate if too long (keep space for boundary tokens)
            max_content_length = (
                self.chunk_size - 2
            )  # Reserve space for boundary tokens

            if len(content_tokens) > max_content_length:
                content_tokens = content_tokens[:max_content_length]

            # Add boundary tokens: <chunk_start> + content + <chunk_end>
            chunk_tokens = torch.cat(
                [
                    torch.tensor([chunk_start_id]),
                    content_tokens,
                    torch.tensor([chunk_end_id]),
                ]
            )

            # Pad to exactly chunk_size tokens if needed
            if len(chunk_tokens) < self.chunk_size:
                padding_length = self.chunk_size - len(chunk_tokens)
                pad_token_id = self.tokenizer.pad_token_id
                padding = torch.full(
                    (padding_length,), pad_token_id, dtype=chunk_tokens.dtype
                )
                chunk_tokens = torch.cat([chunk_tokens, padding])

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones_like(chunk_tokens)
            real_content_length = 1 + len(content_tokens) + 1  # start + content + end
            if real_content_length < self.chunk_size:
                attention_mask[real_content_length:] = 0

            # Decode chunk text for verification
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunk_dict = {
                "chunk_id": chunk_id,
                "input_ids": chunk_tokens,
                "attention_mask": attention_mask,
                "text": chunk_text,
                "source_title": source_info["title"],
                "source_context_idx": source_info["context_idx"],
                "center_sentence_idx": window["center_sentence_idx"],
                "center_sentence_text": window["center_sentence_text"],
                "sentence_indices": window[
                    "sentence_indices"
                ],  # All sentences in window
                "window_sentences": window["sentences"],  # Actual sentence texts
            }

            chunks.append(chunk_dict)

        return chunks

    def find_gold_chunks(
        self,
        chunks: List[Dict],
        supporting_facts: List[List],
        original_passages: List[Dict[str, str]],
    ) -> List[int]:
        """
        Find which chunks contain gold supporting facts using sentence windows.

        Args:
            chunks: List of chunk dictionaries with center_sentence_idx
            supporting_facts: List of [title, sentence_idx] pairs from HotpotQA
            original_passages: List of original passage dictionaries for sentence lookup

        Returns:
            List of chunk IDs that contain supporting facts as center sentences
        """
        gold_chunk_ids = []

        # For each supporting fact, find chunks where it's the center sentence
        for title, sentence_idx in supporting_facts:
            print(f"    Looking for title '{title}', sentence {sentence_idx}")

            # Find chunks from this title where this sentence is the center
            title_chunks = [c for c in chunks if c["source_title"] == title]
            print(f"    Title '{title}': found {len(title_chunks)} chunks")

            found_match = False

            for chunk in title_chunks:
                # Check if this sentence is the center sentence of this chunk
                if chunk.get("center_sentence_idx") == sentence_idx:
                    print(
                        f"    MATCH: chunk {chunk['chunk_id']} has sentence {sentence_idx} as center"
                    )
                    if chunk["chunk_id"] not in gold_chunk_ids:
                        gold_chunk_ids.append(chunk["chunk_id"])
                    found_match = True
                else:
                    center_idx = chunk.get("center_sentence_idx", -1)
                    print(
                        f"    No match in chunk {chunk['chunk_id']}: center sentence is {center_idx}"
                    )
                    print(
                        f"    Chunk center text: '{chunk.get('center_sentence_text', '')[:100]}...'"
                    )

            if not found_match and title_chunks:
                print(f"    DEBUG: Original sentences for '{title}':")
                title_passage = next(
                    (p for p in original_passages if p["title"] == title), None
                )
                if title_passage:
                    for i, sent in enumerate(
                        title_passage.get("original_sentences", [])
                    ):
                        marker = " <-- SUPPORTING FACT" if i == sentence_idx else ""
                        print(f"      Sentence {i}: '{sent[:100]}...'{marker}")
                else:
                    print(f"      No passage found for title '{title}'")

            if not found_match and title_chunks:
                print(
                    f"    WARNING: No chunks found with sentence {sentence_idx} as center for title '{title}'"
                )
            elif not title_chunks:
                print(f"    No chunks found for title '{title}'")

        return gold_chunk_ids

    def process_hotpotqa_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a HotpotQA JSON file into chunks and examples.

        Returns:
            chunks: List of all chunks with tokenized content
            examples: List of examples with chunk indices and gold mappings
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        all_chunks = []
        examples = []
        chunk_id_counter = 0

        print(f"Processing {len(data)} examples from {file_path}...")

        for example_idx, example in enumerate(tqdm(data)):

            # Extract passages from context
            passages = self.extract_passages_from_context(example["context"])

            # Create chunks from all passages in this example
            example_chunks = []

            for passage in passages:

                source_info = {
                    "title": passage["title"],
                    "context_idx": passage["context_idx"],
                    "original_sentences": passage.get("original_sentences", []),
                }

                passage_chunks = self.create_chunks_from_sentences(source_info)

                # Add global chunk IDs
                for chunk in passage_chunks:
                    chunk["chunk_id"] = chunk_id_counter  # Global unique ID
                    all_chunks.append(chunk)
                    example_chunks.append(chunk)
                    chunk_id_counter += 1

            # Find gold chunks that contain supporting facts
            gold_chunk_ids = self.find_gold_chunks(
                example_chunks, example["supporting_facts"], passages
            )

            # Debug: print supporting facts and found chunks
            print(
                f"Example {example['_id']}: supporting_facts = {example['supporting_facts']}"
            )
            print(f"  Found {len(gold_chunk_ids)} gold chunks: {gold_chunk_ids}")
            if example_chunks:
                print(f"  Total chunks in example: {len(example_chunks)}")
                all_titles = list(set([c["source_title"] for c in example_chunks]))
                print(f"  All chunk titles in example: {all_titles}")
                supporting_titles = [sf[0] for sf in example["supporting_facts"]]
                print(f"  Supporting fact titles: {supporting_titles}")
                print(f"  Titles overlap: {set(supporting_titles) & set(all_titles)}")

            # Skip examples with no gold chunks found
            if not gold_chunk_ids:
                continue

            # Get chunk IDs for this example
            example_chunk_ids = [chunk["chunk_id"] for chunk in example_chunks]

            # Tokenize the question - pad to length of n_docs * chunk_size
            # This will be reshaped to [n_docs, chunk_size] in dataset
            question_max_length = self.n_docs * self.chunk_size
            question_tokens = self.tokenizer(
                example["question"],
                max_length=question_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize the answer
            answer_tokens = self.tokenizer(
                example["answer"],
                max_length=32,  # Configurable target length
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            example_entry = {
                "question_id": example["_id"],
                "question": example["question"],
                "question_tokens": question_tokens["input_ids"].squeeze(0).tolist(),
                "question_attention_mask": question_tokens["attention_mask"]
                .squeeze(0)
                .tolist(),
                "answer": example["answer"],
                "answer_tokens": answer_tokens["input_ids"]
                .squeeze(0)
                .tolist(),  # Convert to list for JSON
                "answer_attention_mask": answer_tokens["attention_mask"]
                .squeeze(0)
                .tolist(),
                "chunk_ids": example_chunk_ids,
                "gold_chunk_ids": gold_chunk_ids,
                "level": example.get("level", "unknown"),
                "type": example.get("type", "unknown"),
                "num_chunks": len(example_chunks),
            }

            examples.append(example_entry)

        print(f"Processed {len(examples)} examples, {len(all_chunks)} total chunks")
        if len(examples) > 0:
            print(f"Average chunks per example: {len(all_chunks)/len(examples):.1f}")
        else:
            print("Warning: No examples with gold chunks found")
        return all_chunks, examples

    def save_processed_data(
        self, chunks: List[Dict], examples: List[Dict], output_dir: Path
    ):
        """Save processed chunks and examples to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_file = output_dir / "chunks.pt"
        torch.save(chunks, chunks_file)

        # Save examples
        examples_file = output_dir / "examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples, f, indent=2)

        print(
            f"Saved {len(chunks)} chunks and {len(examples)} examples to {output_dir}"
        )

        return chunks_file, examples_file


def preprocess_hotpotqa_dataset(
    raw_data_path: Path,
    output_dir: Path,
    pretrained_model_name: str = "gpt2",
    chunk_size: int = 256,
    overlap_size: int = 32,
    n_docs: int = 8,
):
    """
    Main function to preprocess HotpotQA dataset into fixed-size chunks.

    Args:
        raw_data_path: Path to HotpotQA JSON file
        output_dir: Output directory for processed data
        pretrained_model_name: Model name for tokenizer
        chunk_size: Size of each chunk in tokens (default: 256)
        overlap_size: Overlap between chunks in tokens (default: 32)
    """
    preprocessor = HotpotQAPreprocessor(
        pretrained_model_name=pretrained_model_name,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        n_docs=n_docs,
    )

    chunks, examples = preprocessor.process_hotpotqa_file(raw_data_path)
    chunks_file, examples_file = preprocessor.save_processed_data(
        chunks, examples, output_dir
    )

    return chunks_file, examples_file


if __name__ == "__main__":
    # Example usage
    raw_data_path = Path("data/hotpotqa/hotpot_train_v1.1.json")
    output_dir = Path("data/processed/hotpotqa_train")

    preprocess_hotpotqa_dataset(raw_data_path, output_dir)
