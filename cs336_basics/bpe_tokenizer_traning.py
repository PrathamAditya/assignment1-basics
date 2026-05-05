import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def bpe_merge(tuple_list, next_id, vocab, merges):
    pair_frequency_dict = {}
    word_dict = {}
    
    for tup in tuple_list:
        if tup in word_dict:
            word_dict[tup] += 1
        else:
            word_dict[tup] = 1

    # computing pair frequency
    for word in word_dict:
        for i in range(0, len(word) -1 ):
            temp_tuple = (word[i], word[i+1])
            pair_frequency_dict[temp_tuple] = pair_frequency_dict.get(temp_tuple, 0) + word_dict[word]

    if not pair_frequency_dict:
        return tuple_list, next_id
    
    best_pair = max(pair_frequency_dict, key=lambda pair: (pair_frequency_dict[pair], pair))
    print(f"XOXO{best_pair}XOXO")
    a, b = best_pair
    vocab[next_id] = vocab[a] + vocab[b]
    merges.append((vocab[a], vocab[b]))
    new_tuple_list = []

    for tup in tuple_list:
        new_word = []
        i = 0
        while i < len(tup):
            if i < len(tup) - 1 and (tup[i], tup[i+1]) == best_pair:
                new_word.append(next_id)
                i += 2
            else:
                new_word.append(tup[i])
                i += 1
        new_tuple_list.append(tuple(new_word))
    next_id += 1
    return new_tuple_list, next_id

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
    next_id = add_special_token(special_tokens, next_id, vocab)

    # pre-tokenization
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(pattern, file_content)
    else:
        chunks = [file_content]

    for chunk in chunks:
        pre_tokens = re.findall(PAT, chunk)
        for token in pre_tokens:
            byte_tuple_list.append(tuple(token.encode("utf-8")))
        if chunk_counter % 1000 == 0:
            print(f"chunk: {chunk_counter}")
        chunk_counter += 1

    tokens = byte_tuple_list
    while len(vocab) < vocab_size:
        new_tokens, next_id = bpe_merge(tokens, next_id, vocab, merges)

        if(new_tokens == tokens):
            break
        
        tokens = new_tokens
    
    return vocab, merges

if __name__=="__main__":

    file_path = f'../data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 1000
    special_token_list = ["<|endoftext|>"]

    v, m = train_bpe(file_path, vocab_size, special_token_list)

    with open("../data/training_output/vocab.txt", "w", encoding='utf-8') as f:
        for k, v_bytes in v.items():
            f.write(f"{k}: {v_bytes}\n")

    with open("../data/training_output/merge.txt", "w", encoding='utf-8') as f:
        for a, b in m:
            f.write(f"{a} {b}\n")

