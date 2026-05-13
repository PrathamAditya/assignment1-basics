def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# text = "Hello hello こんにちは"
# text = "143"
text = "  "
# encoded_string_8 = text.encode("utf-8")
encoded_string_8 = [0, 0]
# encoded_string_16 = text.encode("utf-16")
# encoded_string_32 = text.encode("utf-32")

print(bytes([194, 32]).decode())
# print(list(encoded_string_16))
# print(list(encoded_string_32))
# print(decode_utf8_bytes_to_str_wrong(encoded_string_8))
