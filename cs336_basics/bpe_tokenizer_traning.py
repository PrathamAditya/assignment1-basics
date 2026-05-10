import regex as re
from collections import defaultdict
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def bpe_merge(tuple_list, next_id, vocab, merges, pair_frequency_dict, word_dict, pair_to_words_dict):
   
    if not pair_frequency_dict:
        return tuple_list, next_id
    
    best_pair = max(pair_frequency_dict, key=lambda pair: (pair_frequency_dict[pair], pair))
    old_list = list(pair_to_words_dict[best_pair])
    # distinct_pairs = set()
    for word in old_list:
        freq = word_dict[word]
        
        for i in range(0, len(word)-1):
            temp_tuple = (word[i], word[i+1])
            # distinct_pairs.add(temp_tuple)
            pair_frequency_dict[temp_tuple] -= freq
            # if pair_frequency_dict[temp_tuple] == 0:
            #     del pair_frequency_dict[temp_tuple]

        # for item in distinct_pairs:
        #     if not (item in pair_frequency_dict):
        #         pair_to_words_dict[item].remove(word)

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
    if next_id % 100 == 0:
        print(f"XOXO{next_id, best_pair}XOXO")
    a, b = best_pair
    vocab[next_id] = vocab[a] + vocab[b]
    merges.append((vocab[a], vocab[b]))
    next_id += 1
    return next_id

def add_special_token(special_tokens, next_id, vocab):
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return next_id

def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str]):
    next_id = 256
    byte_tuple_list = []
    chunk_counter = 1
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    pair_frequency_dict = {}
    word_dict = {}
    pair_to_words_dict = defaultdict(set)
    next_id = add_special_token(special_tokens, next_id, vocab)

    # pre-tokenization
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(pattern, file_content)
    else:
        chunks = [file_content]

    # for test
    # chunks = ["abbaaababa, bababbababab", "aaaaaaaaaa", "Pratham", "Aditya", "Salhotras"]
    # chunks = ["aaaaabaaaaa", "Pratham"]
    # chunks = ["abab"]
    # chunks = ["aaaa"]
    # chunks = [
    #             "abab",
    #             "aaaa",
    #             "abcabc",
    #             "that",
    #             "there",
    #             "this",
    #             "tr tr",
    #             "abababab"]

    for chunk in chunks:
        pre_tokens = re.findall(PAT, chunk)
        # print(pre_tokens)
        for token in pre_tokens:
            byte_tuple_list.append(tuple(token.encode("utf-8")))
        # if chunk_counter > 999:
        #     break
        # if chunk_counter > 10000: break
        if chunk_counter % 1000 == 0:
            print(f"chunk: {chunk_counter}")
        chunk_counter += 1

    tokens = byte_tuple_list

    for tup in tokens:
        if tup in word_dict:
            word_dict[tup] += 1
        else:
            word_dict[tup] = 1

    for word, freq in word_dict.items():
        for i in range(0, len(word)-1 ):
            temp_tuple = (word[i], word[i+1])
            pair_frequency_dict[temp_tuple] = pair_frequency_dict.get(temp_tuple, 0) + freq
            pair_to_words_dict[temp_tuple].add(word)
    
 
    while len(vocab) < vocab_size:
        next_id = bpe_merge(tokens, next_id, vocab, merges, pair_frequency_dict, word_dict, pair_to_words_dict)
        # if(new_tokens == tokens):
        #     break
        # tokens = new_tokens
    return vocab, merges

if __name__=="__main__":
    #file_path = f'assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    file_path = f'../data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_token_list = ["<|endoftext|>"]

    v, m = train_bpe(file_path, vocab_size, special_token_list)
    # print(m)
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

