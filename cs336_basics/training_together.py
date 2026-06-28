import numpy as np
import torch
from .cross_entropy import CrossEntropy 
from .gradient_clipping import GradientClipping
from .checkpointing import save_checkpoint
from .learning_rate_schedule import LearningRateSchedule
from .data_loading import DataLoading, DataLoadingValidationSequential
from .transformer_lm import TransformerLM
from .adamw import AdamW
from .gradient_clipping import GradientClipping
import time
import cProfile
import pstats

def train(model: TransformerLM, optimizer, train_data, batch_size, context_length, 
          max_iterations, device: torch.device, checkpoint_path, val_data = None):
    # Learning-rate schedule
    warmup_steps = 400
    cosine_end_steps = max_iterations
    alpha_max = 6e-4
    alpha_min = 5e-5
    eval_interval = 100
    checkpoint_interval = 1500
    max_l2_norm = 1.0
    model.train()

    tokens_per_step = batch_size * context_length
    torch.cuda.synchronize()
    throughput_start = time.perf_counter()
    validation_counter = 0
    for iteration in  range(1, max_iterations + 1):
        lr = LearningRateSchedule(alpha_max=alpha_max, alpha_min=alpha_min, t=iteration, T_w=warmup_steps, T_c=cosine_end_steps)
        if iteration % eval_interval == 0:
            validation_counter += 1
            model.eval() 
            total_val_loss = 0.0
            val_batches = 0
        
            with torch.no_grad():
            
                for x_val, y_val in DataLoadingValidationSequential(val_data, batch_size, context_length, device=device):
                    logits = model(x_val)
                    loss = CrossEntropy(logits, y_val)

                    total_val_loss += loss.item()
                    val_batches += 1
        
            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
            print(f"### Validation at {iteration}: Avg loss: {avg_val_loss:.4f} #####")
            model.train()

        x, y = DataLoading(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = CrossEntropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        GradientClipping(model.parameters(), max_l2_norm)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr


        if iteration%checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
        # if iteration % 50 == 0 and iteration%checkpoint_interval != 0:
        if iteration % 50 == 0:
            print(f"Iteration={iteration}, LR={lr:.2e}, Loss={loss.item():.4f}")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - throughput_start
            tokens_processed = 50 * tokens_per_step
            tokens_per_sec = tokens_processed / elapsed
            print(f"Avg Tokens/sec (last 50 steps): {tokens_per_sec:.2f}")
            throughput_start = time.perf_counter()


def main():
    checkpoint_path = "checkpoints/model_280626_10000.pth"
    # Modal hyperparameters
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 6
    context_length = 256
    # Training hyperparameters
    batch_size = 8
    learning_rate = 6e-4
    max_iterations = 3000
    weight_decay = 0.01
    theta = 10000.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab_size = 10000
    train_data = np.load("data/traning_output/train_tokens.npy", mmap_mode='r')
    valid_data = np.load("data/traning_output/valid_tokens.npy", mmap_mode='r')
    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta).to(device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8 )
    train(model, optimizer, train_data, batch_size, context_length, max_iterations, device, checkpoint_path, val_data=valid_data)



if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumtime') # Sort by cumulative time
    # stats.print_stats()

