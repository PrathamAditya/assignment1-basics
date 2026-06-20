# import regex as re
# import numpy as np
# import ast
# from cs336_basics.tokenizer import Tokenizer
# import os
# # # import Tokenizer
# # # endcode text
# # # step 1: pre tokenize
# # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}+|\s+(?!\S)|\s+"""
# # compiled_pat = re.compile(PAT)
# # input_text = "This is sample text, I am using. 1"
# # input_text = compiled_pat.finditer(input_text)
# # encoded = []


# # for word in input_text:
# #     encoded.append(tuple(word.group().encode("utf-8")))

# # with open(r"D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output\merge.txt", "r", encoding= "utf-8") as merge:
# #     i = 0
# #     merge_output: list[tuple[bytes, bytes]] = []
# #     pattern = r"(b'.*?')"
# #     for voc in merge:
# #         result = re.split(pattern, voc, maxsplit=1)
# #         print(voc)
# #         print(result)
# #         byte_left = ast.literal_eval(result[1])
# #         byte_right = ast.literal_eval(result[2])
# #         merge_output.append((byte_left, byte_right))
# #         # pattern = r'[\s]+'
# #         # result = re.split(pattern, voc, maxsplit=1)
# #         # print(ast.literal_eval(result[0]), ast.literal_eval(result[1]))
# #         # merge_output.append((result[0], result[1]))
# #         # merge_output[int(result[0])] = ast.literal_eval(result[1].strip())
# #         i+=1
# #         if i >= 50: break

# # print(merge_output)
#         # vocab
# # with open(r"D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output\vocab.txt", "r", encoding = "utf-8") as v:
# #     pattern = r'[:\s]+'
# #     i=1
# #     local_vocab = {}
# #     for voc in v:
# #         result = re.split(pattern, voc, maxsplit=1)
# #         local_vocab[int(result[0])] = ast.literal_eval(result[1].strip())
# #         i+=1
# #         if i >= 5: break 
# #     print(local_vocab)
# # temp_dict = {"key1": "Value1", "key2": "Value2"}
# # for xyz in temp_dict:
# #     # k, v = xyz
# #     print(f"{xyz}: {temp_dict[temp_dict]}")

# # text = "hello"

# # # best_pair = ()
# # # if not best_pair:
# # #     print("Best Pair is empty.")

# # bytes_obj_list = [bytes([b]) for b in text.encode("utf-8")]
# # for item in bytes_obj_list:
# #     print(item)
# # print(bytes_obj_list)
# test_strings_1 = [
#     "",
#     " ",
#     "    ",
#     "\n",
#     "\t",

#     "a",
#     "ab",
#     "abc",
#     "abab",
#     "aaaa",
#     "aaaaaaaaaa",
#     "ababababab",
#     "abcabcabcabc",

#     "banana",
#     "mississippi",

#     " hello",
#     "hello world",
#     " the cat sat",
#     "the    cat",
#     "a b c d",

#     "hello!",
#     "wait...",
#     "what?!",
#     "a-b-c",
#     "can't",
#     "i'll",

#     "123",
#     "2026",
#     "1 2 3 4",
#     "abc123",

#     "牛",
#     "こんにちは",
#     "🙂",
#     "🙂🙂🙂",
#     "café",
#     "naïve",

#     "hello 牛 world",
#     "🙂 hello 🙂",
#     "भारत",
#     "مرحبا",

#     "hello\nworld",

#     "antidisestablishmentarianism",
#     "supercalifragilisticexpialidocious",

#     "<|endoftext|>",
#     "hello<|endoftext|>world",
# ]

# # test_strings_2 = ["é", "こんにちは", "<|endoftext|>"]

# # input_strings = ["a", "hello", "hello world", "hi\nthere\t!", "café", "hello 😊","Hello नमस्ते 你好"]
# i_will_catch_you = Tokenizer.from_files(r"../data/training_output _original/vocab.txt", 
#                                         r"../data/training_output _original/merge.txt")


# with open(r"../data/TinyStories10.txt", "r", encoding = "utf-8") as t:
#     tokens = i_will_catch_you.encode_iterable(t)
#     with open(r"../data/TinyStories10-Tokens.txt", "w") as file:
#         for line in tokens:
#             file.write(f"{line}\n")


# # Compression ratio

# # 1. Raw input size from metadata
# bytes_input = os.path.getsize(r"../data/TinyStories10.txt")

# # 2. Memory-efficient token counting
# with open(r"../data/TinyStories10-Tokens.txt", "r", encoding="utf-8") as f:
#     num_tokens = sum(len(line.split()) for line in f)

# # 3. Calculate metrics assuming 4 bytes per standard 32-bit int token
# bytes_per_token = bytes_input / num_tokens
# compression_ratio = (num_tokens * 2) / bytes_input

# # --- Output ---
# print(f"Input Bytes:       {bytes_input}")
# print(f"Total Tokens:      {num_tokens}")
# print(f"Bytes / Token:     {bytes_per_token:.2f}")
# print(f"True Comp Ratio:   {compression_ratio:.2f}")
# # for str in test_strings_2:
# #     print(i_will_catch_you.encode(str))
# #     print(i_will_catch_you.decode(i_will_catch_you.encode(str)))

# # text = "<|endoftext|>"
# # print(text.encode("utf-8"))
# # # print([bytes([text.encode("utf-8")])])
##############################################################################################################

import torch

# tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
logits = torch.tensor([
    [
        [2.0, 5.0, 1.0, 3.0, 4.0],
        [1.0, 4.0, 2.0, 5.0, 3.0],
        [3.0, 1.0, 5.0, 2.0, 4.0]
    ],
    [
        [4.0, 2.0, 1.0, 5.0, 3.0],
        [5.0, 1.0, 3.0, 2.0, 4.0],
        [1.0, 3.0, 4.0, 2.0, 5.0]
    ]
])

target_token_ids = torch.tensor([
    [1, 3, 2],
    [3, 0, 4]
])
temp_tensor = torch.max(logits, dim = -1)
temp_tensor = temp_tensor.values.unsqueeze(-1)
logits = logits - temp_tensor
exp_logits = torch.exp(logits) # exp last dim
sum_exp = torch.sum(exp_logits, dim=-1)
log_logits = torch.log(sum_exp)
o_result = torch.gather(logits, dim=2, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
print(log_logits.shape)
print(o_result.shape)
result = log_logits - o_result
print(result)