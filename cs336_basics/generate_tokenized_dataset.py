import os
import struct
import numpy as np
from .tokenizer import Tokenizer

def make_safe_file_stream(filepath: str):
    """
    Reads a massive file line-by-line and yields text with consistent spacing,
    preventing massive RAM spikes.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip() + " "

def main():
    vocab_path = r"data/training_output_original/vocab.txt"
    merge_path = r"data/training_output_original/merge.txt"
    input_txt = "data/TinyStoriesV2-GPT4-valid.txt"
    
    output_dir = "data/training_output/train_tokens"
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "toystoryvalid.bin")
    npy_path = os.path.join(output_dir, "train_tokens.npy")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_path, merge_path,
    special_tokens=["<|endoftext|>"])

    print(f"Streaming and tokenizing: {input_txt}")
    file_stream = make_safe_file_stream(input_txt)
    token_iterator = tokenizer.encode_iterable(file_stream)

    print(f"Writing tokens to binary: {bin_path}")
    chunk_size = 1_000_000
    chunk_buffer = []

    with open(bin_path, "wb") as f_out:
        for token_id in token_iterator:
            chunk_buffer.append(token_id)
            if len(chunk_buffer) >= chunk_size:
                f_out.write(struct.pack(f'{len(chunk_buffer)}I', *chunk_buffer))
                chunk_buffer.clear()
        
        if chunk_buffer:
            f_out.write(struct.pack(f'{len(chunk_buffer)}I', *chunk_buffer))

    print(f"Converting binary to NumPy format: {npy_path}")
    token_array = np.fromfile(bin_path, dtype=np.uint32)
    np.save(npy_path, token_array)
    
    print("Dataset tokenization complete!")

if __name__ == "__main__":
    main()