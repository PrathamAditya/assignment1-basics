import regex as re
import numpy as np
import ast
# import Tokenizer
# endcode text
# step 1: pre tokenize
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_pat = re.compile(PAT)
input_text = "This is sample text, I am using. 1"
input_text = compiled_pat.finditer(input_text)
encoded = []

# for word in input_text:
#     encoded.append(tuple(word.group().encode("utf-8")))

# with open(r"D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output\merge.txt", "r", encoding= "utf-8") as merge:
#     i = 0
#     merge_output: list[tuple[bytes, bytes]] = []
#     pattern = r"(b'.*?')"
#     for voc in merge:
#         result = re.split(pattern, voc, maxsplit=1)
#         print(voc)
#         print(result)
#         byte_left = ast.literal_eval(result[1])
#         byte_right = ast.literal_eval(result[2])
#         merge_output.append((byte_left, byte_right))
#         # pattern = r'[\s]+'
#         # result = re.split(pattern, voc, maxsplit=1)
#         # print(ast.literal_eval(result[0]), ast.literal_eval(result[1]))
#         # merge_output.append((result[0], result[1]))
#         # merge_output[int(result[0])] = ast.literal_eval(result[1].strip())
#         i+=1
#         if i >= 50: break

# print(merge_output)
        # vocab
# with open(r"D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output\vocab.txt", "r", encoding = "utf-8") as v:
#     pattern = r'[:\s]+'
#     i=1
#     local_vocab = {}
#     for voc in v:
#         result = re.split(pattern, voc, maxsplit=1)
#         local_vocab[int(result[0])] = ast.literal_eval(result[1].strip())
#         i+=1
#         if i >= 5: break 
#     print(local_vocab)
temp_dict = {"key1": "Value1", "key2": "Value2"}
for xyz in temp_dict:
    # k, v = xyz
    print(f"{xyz}: {temp_dict[temp_dict]}")