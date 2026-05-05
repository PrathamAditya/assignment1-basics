import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
word_dict = {}
byte_tuple_list = []
current_token_id = 256
str = "some text that i'll pre-tokenize some text is goint to repeat because need for testinghehe."

def bpe_merge(tuple_list: list):
    pair_frequency_dict = {}
    merge_rule = {}
    

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
    
    best_pair = max(pair_frequency_dict, key=lambda pair: (pair_frequency_dict[pair], pair))
    merge_rule[best_pair] = current_token_id
    current_token_id += 1

    # here comes the loop to merge over all the tokens


    return pair_frequency_dict, merge_rule

if __name__=="__main__":
    # pre-tokenization
    pre_tokens = re.findall(PAT, str)
    
    # byte tuples from string tokens
    for token in pre_tokens:
        byte_tuple_list.append(tuple(token.encode("UTF-8")))
    
    a, b = bpe_merge(byte_tuple_list)
    

