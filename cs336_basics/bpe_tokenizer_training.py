import regex as re
from collections import defaultdict
import os
import cProfile
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def bpe_merge(next_id, vocab, merges, pair_frequency_dict, word_dict, pair_to_words_dict):
   
    if not pair_frequency_dict:
        return next_id
    
    best_pair = max(
        pair_frequency_dict, 
        key=lambda pair: (pair_frequency_dict[pair], (vocab[pair[0]], vocab[pair[1]]))
    )
    
    old_list = list(pair_to_words_dict[best_pair])
    for word in old_list:
        freq = word_dict[word]
        
        for i in range(0, len(word)-1):
            temp_tuple = (word[i], word[i+1])
            pair_frequency_dict[temp_tuple] -= freq

    for old_word in old_list:
        new_word = []
        distinct_pairs = set()
        freq = word_dict[old_word]
        j = 0
        while j < len(old_word):
            if j < len(old_word) - 1 and (old_word[j], old_word[j+1]) == best_pair:
                new_word.append(next_id)
                j += 2
            else:
                new_word.append(old_word[j])
                j += 1
        new_word = tuple(new_word)

        for j in range(len(new_word) - 1):
            new_pair = (new_word[j], new_word[j+1])
            
            pair_frequency_dict[new_pair] = pair_frequency_dict.get(new_pair, 0) + freq
            pair_to_words_dict[new_pair].add(new_word)
        
        word_dict[new_word] = word_dict.get(new_word, 0) + freq
        
        if old_word in old_list:
            freq = word_dict[old_word]
            for i in range(0, len(old_word)-1):
                pair = (old_word[i], old_word[i+1])
                distinct_pairs.add(pair)
                if pair in pair_frequency_dict and pair_frequency_dict[pair] == 0:
                    del pair_frequency_dict[pair]
            del word_dict[old_word]

        for pair in distinct_pairs:
            pair_to_words_dict[pair].remove(old_word)
            if pair in pair_frequency_dict and pair_frequency_dict[pair] == 0:
                    del pair_frequency_dict[pair]
            if len(pair_to_words_dict[pair]) == 0:
                del pair_to_words_dict[pair]

    a, b = best_pair
    vocab[next_id] = vocab[a] + vocab[b]
    merges.append((vocab[a], vocab[b]))
    next_id += 1
    return next_id

# def add_special_token(special_tokens, next_id, vocab):
#     for tok in special_tokens:
#         vocab[next_id] = tok.encode("utf-8")
#         next_id += 1
#     return next_id

# def process_batch(chunk_data, pattern):
#     # text = chunk_data.decode('utf-8', errors='ignore')    
#     word_dict = {}

#     for match in re.finditer(pattern, chunk_data):
#         token = match.group()
#         encoded = tuple(token.encode("utf-8"))
        
#         if encoded in word_dict:
#             word_dict[encoded] += 1
#         else: 
#             word_dict[encoded] = 1
    
#     return word_dict
def process_batch(segments, pattern):

    word_dict = Counter()
    compiled_pat = re.compile(pattern)

    for text in segments:
        for match in compiled_pat.finditer(text):
            word = match.group()
            encoded_tuple = tuple(word.encode("utf-8"))
            word_dict[encoded_tuple] += 1
            
    return word_dict

def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str]):

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    merges = []
    pair_frequency_dict = {}
    word_dict = {}
    pair_to_words_dict = defaultdict(set)
    

    if special_tokens:
        pattern_str = "|".join(map(re.escape, special_tokens))
    else:
        pattern_str = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    if pattern_str:
        segments = re.split(pattern_str, data)
    else:
        segments = [data]

    num_workers = 8
    batches = [[] for _ in range(num_workers)]
    for i, seg in enumerate(segments):
        batches[i % num_workers].append(seg)
    # file_size = len(data)
    # target_size = file_size // 8
    # boundaries = [0]
    # curr_pos = 0
    
    # for _ in range(7):
    #     search_start = curr_pos + target_size
    #     match = regex.search(data, search_start)
    #     curr_pos = match.end() if match else curr_pos + target_size
    #     boundaries.append(curr_pos)
    # boundaries.append(file_size)

    total_frequencies = Counter()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch, PAT) for batch in batches]
        
        for future in futures:
            total_frequencies.update(future.result())

    word_dict = total_frequencies
    for word, freq in word_dict.items():
        for i in range(0, len(word)-1 ):
            temp_tuple = (word[i], word[i+1])
            pair_frequency_dict[temp_tuple] = pair_frequency_dict.get(temp_tuple, 0) + freq
            pair_to_words_dict[temp_tuple].add(word)
 
    while len(vocab) < vocab_size:
        temp_id = next_id
        next_id = bpe_merge(next_id, vocab, merges, pair_frequency_dict, word_dict, pair_to_words_dict)
        if(next_id == temp_id):
            break
    return vocab, merges

if __name__=="__main__":
    pr = cProfile.Profile()
    pr.enable()
    # file_path = f'assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    file_path = f'../data/TinyStoriesV2-GPT4-valid.txt'
    # vocab_size = 3000
    vocab_size = 2000
    special_token_list = ["<|endoftext|>"]

    v, m = train_bpe(file_path, vocab_size, special_token_list)
    output_dir = r"D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output"
    os.makedirs(output_dir, exist_ok=True)
    merge_path = os.path.join(output_dir, "merge.txt")
    vocab_path = os.path.join(output_dir, "vocab.txt")
    # with open("assignment1-basics/data/training_output/vocab.txt", "w", encoding='utf-8') as f:
    # with open("../data/training_output/vocab.txt", "w", encoding='utf-8') as f:
    with open(vocab_path, "w", encoding='utf-8') as vocab:
        for k, v_bytes in v.items():
            vocab.write(f"{k}: {v_bytes}\n")

    # with open("assignment1-basics/data/training_output/merge.txt", "w", encoding='utf-8') as f:
    with open(merge_path, "w", encoding='utf-8') as merge:
        for a, b in m:
            merge.write(f"{a} {b}\n")
    
    pr.disable()
    pr.print_stats(sort='time')

