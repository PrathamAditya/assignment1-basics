import regex as re
from collections import defaultdict, Counter
import os
import cProfile
import heapq
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPEMergeQueue:
    """A Priority Queue to find the most frequent pair in O(log N) time instead of O(N)."""
    def __init__(self, pair_frequency_dict, vocab):
        self.heap = []
        self.vocab = vocab
        self.valid_counts = {}
        for pair, freq in pair_frequency_dict.items():
            self.push(pair, freq)
            
    def push(self, pair, freq):
        self.valid_counts[pair] = freq
        # Max-heap in Python uses negative values.
        # Tie-breaker: lexicographical order of bytes.
        tie_breaker = self.vocab[pair[0]] + self.vocab[pair[1]]
        heapq.heappush(self.heap, (-freq, tie_breaker, pair))
        
    def pop_best(self):
        while self.heap:
            neg_freq, _, pair = heapq.heappop(self.heap)
            freq = -neg_freq
            # Only return if it's the most up-to-date frequency (ignores stale heap nodes)
            if self.valid_counts.get(pair, 0) == freq:
                del self.valid_counts[pair]
                return pair, freq
        return None, None
        
    def update(self, pair, freq):
        if freq <= 0:
            self.valid_counts.pop(pair, None)
        else:
            self.push(pair, freq)


def process_chunk(text, pattern_str, special_tokens_str):
    """Processes a chunk of text. Safe for multiprocessing."""
    word_dict = Counter()
    compiled_pat = re.compile(pattern_str)
    
    if special_tokens_str:
        special_pat = re.compile(special_tokens_str)
        segments = special_pat.split(text)
    else:
        segments = [text]
        
    for seg in segments:
        if not seg:
            continue
        for match in compiled_pat.finditer(seg):
            word = match.group()
            word_dict[tuple(word.encode("utf-8"))] += 1
            
    return word_dict


def read_batches(filepath, batch_size=500000):
    """Streams the file in chunks to prevent Out-Of-Memory (OOM) errors."""
    batch = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            batch.append(line)
            if len(batch) >= batch_size:
                yield "".join(batch)
                batch = []
        if batch:
            yield "".join(batch)

def word_to_linked_list(word_tuple):
    nodes = {}
    for i, token in enumerate(word_tuple):
        nodes[i] = {
            'val': token,
            'prev': i - 1 if i > 0 else None,
            'next': i + 1 if i < len(word_tuple) - 1 else None
        }
    return nodes, 0, len(word_tuple) - 1  # nodes, head, tail
def apply_merge(nodes, head, pair, new_id):
    removed = []
    added = []
    i = head
    while i is not None:
        n = nodes[i]
        nxt = n['next']
        if nxt is not None and n['val'] == pair[0] and nodes[nxt]['val'] == pair[1]:
            left_neighbor = n['prev']
            right_neighbor = nodes[nxt]['next']
            if left_neighbor is not None:
                removed.append((nodes[left_neighbor]['val'], pair[0]))
            removed.append((pair[0], pair[1]))
            if right_neighbor is not None:
                removed.append((pair[1], nodes[right_neighbor]['val']))
            n['val'] = new_id
            n['next'] = right_neighbor
            if right_neighbor is not None:
                nodes[right_neighbor]['prev'] = i
            del nodes[nxt]
            if left_neighbor is not None:
                added.append((nodes[left_neighbor]['val'], new_id))
            if right_neighbor is not None:
                added.append((new_id, nodes[right_neighbor]['val']))
            i = right_neighbor
        else:
            i = nxt
    return removed, added

def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    merges = []
    
    special_tokens_str = "|".join(map(re.escape, special_tokens)) if special_tokens else None
    word_dict = Counter()
    # num_workers = max(1, os.cpu_count() - 1)
    num_workers = 4
    
    print(f"[*] Processing dataset using {num_workers} workers...")

    batches = read_batches(file_path)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(process_chunk, batch, PAT, special_tokens_str))
            # Drain when we have enough pending futures
            if len(futures) >= num_workers * 2:
                word_dict.update(futures.pop(0).result())
        for future in futures:
            word_dict.update(future.result())
    print("[*] Initializing Pair Frequencies...")
    
    pair_frequency_dict = {}
    pair_to_words_dict = defaultdict(set)

    for word, freq in word_dict.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            pair_frequency_dict[p] = pair_frequency_dict.get(p, 0) + freq
            pair_to_words_dict[p].add(word)

    word_dict_ll = {}
    for word_tuple, freq in word_dict.items():
        nodes, head, tail = word_to_linked_list(word_tuple)
        word_dict_ll[word_tuple] = (nodes, head, freq)

    pq = BPEMergeQueue(pair_frequency_dict, vocab)
    print(f"[*] Starting merges to reach vocab size {vocab_size}...")
    while next_id < vocab_size:
        best_pair, best_freq = pq.pop_best()
        if best_pair is None:
            break
            
        a, b = best_pair
        vocab[next_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))
        pair_frequency_dict.pop(best_pair, None)
        words_to_process = list(pair_to_words_dict.pop(best_pair, []))
        
        # for word in words_to_process:
        #     freq = word_dict[word]
            
        #     # Remove old pairs associated with the old word
        #     for i in range(len(word) - 1):
        #         p = (word[i], word[i+1])
        #         if p != best_pair:
        #             pair_frequency_dict[p] -= freq
        #             pq.update(p, pair_frequency_dict[p])
        #             if p in pair_to_words_dict:
        #                 pair_to_words_dict[p].discard(word)
        #                 if pair_frequency_dict[p] <= 0:
        #                     pair_frequency_dict.pop(p, None)
        #                     if not pair_to_words_dict[p]:
        #                         pair_to_words_dict.pop(p, None)
                                
        #     # Construct the new merged word
        #     new_word = []
        #     j = 0
        #     while j < len(word):
        #         if j < len(word) - 1 and word[j] == best_pair[0] and word[j+1] == best_pair[1]:
        #             new_word.append(next_id)
        #             j += 2
        #         else:
        #             new_word.append(word[j])
        #             j += 1
        #     new_word = tuple(new_word)
            
        #     # Add new pairs generated by the new word
        #     for i in range(len(new_word) - 1):
        #         p = (new_word[i], new_word[i+1])
        #         pair_frequency_dict[p] = pair_frequency_dict.get(p, 0) + freq
        #         pq.update(p, pair_frequency_dict[p])
        #         pair_to_words_dict[p].add(new_word)
                
        #     # Finalize dictionary updates
        #     del word_dict[word]
        #     word_dict[new_word] = word_dict.get(new_word, 0) + freq
        for word in words_to_process:
            nodes, head, freq = word_dict_ll[word]
            removed, added = apply_merge(nodes, head, best_pair, next_id)
            for p in removed:
                if p != best_pair:
                    pair_frequency_dict[p] = pair_frequency_dict.get(p, 0) - freq
                    pq.update(p, pair_frequency_dict[p])
                    pair_to_words_dict[p].discard(word)
            for p in added:
                pair_frequency_dict[p] = pair_frequency_dict.get(p, 0) + freq
                pq.update(p, pair_frequency_dict[p])
                pair_to_words_dict[p].add(word)
        next_id += 1
        
    return vocab, merges

if __name__=="__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    file_path = os.path.join('data', 'owt_train.txt')
    vocab_size = 32000
    special_token_list = ["<|endoftext|>"]

    # Only run if file exists to prevent hard crash
    if os.path.exists(file_path):
        v, m = train_bpe(file_path, vocab_size, special_token_list)
        
        output_dir = os.path.join("data", "training_output_owt_280626")
        os.makedirs(output_dir, exist_ok=True)
        
        merge_path = os.path.join(output_dir, "merge.txt")
        vocab_path = os.path.join(output_dir, "vocab.txt")
        
        with open(vocab_path, "w", encoding='utf-8') as vocab_file:
            for k, v_bytes in v.items():
                vocab_file.write(f"{k}: {v_bytes}\n")

        with open(merge_path, "w", encoding='utf-8') as merge_file:
            for a, b in m:
                merge_file.write(f"{a} {b}\n")
    else:
        print(f"File not found: {file_path}. Please check your path.")
    
    pr.disable()
    pr.print_stats(sort='time')