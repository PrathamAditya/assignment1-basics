import os
import numpy as np
import multiprocessing as mp
from cs336_basics.tokenizer import Tokenizer  # adjust import to your module path

VOCAB_PATH = "data/training_output_owt/vocab_train_300626.txt"
MERGES_PATH = "data/training_output_owt/merges_train_300626.txt"
INPUT_PATH = "data/owt_valid.txt"
OUTPUT_PATH = "data/owt_valid_ids.npy"
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_PROCESSES = 6

_worker_tokenizer = None


def init_worker():
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)


def find_chunk_boundaries(file_path, num_chunks, split_token=b"<|endoftext|>"):
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

    chunk_size = file_size // num_chunks
    boundaries = [i * chunk_size for i in range(num_chunks)] + [file_size]

    with open(file_path, "rb") as f:
        for i in range(1, len(boundaries) - 1):
            pos = boundaries[i]
            f.seek(pos)
            buf = b""
            while True:
                chunk = f.read(4096)
                if not chunk:
                    boundaries[i] = file_size
                    break
                buf += chunk
                idx = buf.find(split_token)
                if idx != -1:
                    boundaries[i] = pos + idx
                    break
                pos += len(chunk)

    return sorted(set(boundaries))


def encode_chunk(args):
    start, end = args
    with open(INPUT_PATH, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = _worker_tokenizer.encode(text)
    return np.array(ids, dtype=np.uint16)


def main():
    boundaries = find_chunk_boundaries(INPUT_PATH, NUM_PROCESSES * 4)
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    print(f"Processing {len(chunks)} chunks across {NUM_PROCESSES} processes...")

    with mp.Pool(NUM_PROCESSES, initializer=init_worker) as pool:
        results = pool.map(encode_chunk, chunks)

    all_ids = np.concatenate(results)
    print(f"Total tokens: {len(all_ids)}")
    np.save(OUTPUT_PATH, all_ids)


if __name__ == "__main__":
    # main()
    import numpy as np
    ids = np.load("data/owt_train_ids.npy")
    print(f"Shape: {ids.shape}, dtype: {ids.dtype}")
    print(f"Total tokens: {len(ids):,}")
    print(f"Bytes/token ratio: {11200_000_000 / len(ids):.2f}")
    print(f"Max token id: {ids.max()} (should be < 32000)")
    print(f"First 20: {ids[:20]}")

# import time
# from cs336_basics.tokenizer import Tokenizer

# t = Tokenizer.from_files(
#     "data/training_output_owt/vocab_train_300626.txt",
#     "data/training_output_owt/merges_train_300626.txt",
#     ["<|endoftext|>"]
# )
# with open("data/owt_train.txt", "r", encoding="utf-8") as f:
#     text = f.read(50_000_000)  # 50MB

# start = time.time()
# ids = t.encode(text)
# elapsed = time.time() - start

# mb_per_sec = 50 / elapsed
# print(f"{mb_per_sec:.2f} MB/s single-threaded")
# print(f"Est. total time (6 cores): {11200 / (mb_per_sec * 6) / 60:.1f} min")
# print(f"First 20 ids: {ids[:20]}")