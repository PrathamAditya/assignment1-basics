from .checkpointing import load_checkpoint
from .transformer_lm import TransformerLM
from .tokenizer import Tokenizer
from .adamw import AdamW
from .softmax import Softmax
import torch
import os


def Decoding(model: TransformerLM, temperature, device):
    flag = 1
    max_new_tokens = 312
    tokenizer = Tokenizer.from_files(r"data/training_output_original/vocab.txt", 
                                     r"data/training_output_original/merge.txt", special_tokens = ["<|endoftext|>"])
    special_token_id = tokenizer.encode("<|endoftext|>")
    # print(tokenizer.encode("Hello\n<|endoftext|>\nWorld"))
    # print("EOT ID:", special_token_id[0])
    # print("Bytes:", tokenizer.vocab[special_token_id[0]])
    # print("Decode:", tokenizer.decode([special_token_id[0]]))
    with torch.no_grad():
        model.eval()
        while(flag):
            prompt = input("Enter your prompt: ")
            prompt_tokens = tokenizer.encode(prompt)
            prompt_tokens_tensor = torch.Tensor(prompt_tokens).unsqueeze(0).to(device=device).long()
            last_context_length_tensor = prompt_tokens_tensor
            token_counter = 0
            while token_counter < max_new_tokens:
                logits = model(last_context_length_tensor)
                last_logits = logits[0][-1]
                scaled_logits = last_logits / temperature
                result_after_softmax = Softmax(scaled_logits, -1)
                sorted, indices = torch.sort(result_after_softmax, descending=True)
                cumulative = torch.cumsum(sorted, dim = -1)
                index = -1
                for i in cumulative:
                    index += 1
                    if cumulative[index] >= .90:
                        break
                cumulative = cumulative[0: index+1]
                sorted = sorted[0: index+1]
                indices = indices[0: index+1]
                # renormalize
                sorted = sorted/torch.sum(sorted)
                sampled_token = indices[torch.multinomial(sorted, num_samples=1, replacement=False)].unsqueeze(0).to(device=device)
                # print("sampled:", sampled_token.item(), tokenizer.decode([sampled_token.item()]))
                if sampled_token.item() == 256:
                    print(f"######################{sampled_token.item()}")
                    break
                token_counter += 1
                prompt_tokens_tensor = torch.cat((prompt_tokens_tensor, sampled_token), dim=-1)
                last_context_length_tensor = torch.cat((last_context_length_tensor, sampled_token), dim=-1)
                if(last_context_length_tensor.size(-1) > 256):
                    last_context_length_tensor = last_context_length_tensor[:, -256:]
                
            # print("EOT ID:", special_token_id[0])
            # print(prompt_tokens_tensor[0, -30:].tolist())
            print("=======================================================")
            a_list = prompt_tokens_tensor.cpu().tolist()
            print(" ")
            print(tokenizer.decode(a_list[0]))

            flag = int(input("Want to continue 0 or 1: "))


if __name__=="__main__":
    checkpoint_path = "checkpoints/model_280626_10000.pth"
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 6
    context_length = 256
    batch_size = 8
    learning_rate = 6e-4
    max_iterations = 3000
    weight_decay = 0.01
    theta = 10000.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab_size = 10000
    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta).to(device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8 )
    iteration_num = load_checkpoint(checkpoint_path, model, optimizer)
    Decoding(model, 0.3,device)

