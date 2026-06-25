import numpy as np
import torch
from .cross_entropy import CrossEntropy 
from .gradient_clipping import GradientClipping
from .checkpointing import save_checkpoint
from .learning_rate_schedule import LearningRateSchedule
from .data_loading import DataLoading
from .transformer_lm import TransformerLM
from .adamw import AdamW
from .gradient_clipping import GradientClipping

def train(model: TransformerLM, optimizer, train_data, batch_size, context_length, 
          max_iterations, device: torch.device, checkpoint_path, val_data = None):
    # Learning-rate schedule
    warmup_steps = 500
    cosine_end_steps = max_iterations
    alpha_max = 3e-4
    alpha_min = 3e-5
    eval_interval = 100
    checkpoint_interval = 500
    max_l2_norm = 1.0
    model.train()
    print("XOXO I am here XOXO!")
    for iteration in  range(1, max_iterations + 1):
        lr = LearningRateSchedule(alpha_max=alpha_max, alpha_min=alpha_min, t=iteration, T_w=warmup_steps, T_c=cosine_end_steps)
        x, y = DataLoading(train_data, batch_size, context_length, device)
        # print(x.dtype)
        # print(x.shape)
        logits = model(x)
        loss = CrossEntropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        GradientClipping(model.parameters(), max_l2_norm)
        optimizer.step()
        # for group in optimizer.param_groups:
        #     group["lr"] = lr
        if iteration%checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
        if(iteration%10 == 0):
            print(f"LR={lr:.2e}, Loss={loss.item():.4f}")


def main():
    checkpoint_path = "checkpoints/model_240626_10000.pth"

    # Modal hyperparameters
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4
    context_length = 256

    # Training hyperparameters
    batch_size = 8
    learning_rate = 3e-4
    max_iterations = 10000
    weight_decay = 0.01

    theta = 10000.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab_size = 10000
    train_data = np.load("data/traning_output/train_tokens.npy", mmap_mode='r')
    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta).to(device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8 )
    train(model, optimizer, train_data, batch_size, context_length, max_iterations, device, checkpoint_path)


if __name__ == "__main__":
    main()
