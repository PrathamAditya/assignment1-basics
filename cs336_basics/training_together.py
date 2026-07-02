import numpy as np
import torch
from .cross_entropy import CrossEntropy 
# from .gradient_clipping import GradientClipping
from .checkpointing import save_checkpoint_dual
from .learning_rate_schedule import TrapezoidalLearningRateSchedule
from .data_loading import DataLoading, DataLoadingValidationSequential
from .transformer_lm import TransformerLM
from .adamw import AdamW
from .gradient_clipping import GradientClipping
import time
from .experiment_logger import ExperimentLogger
from .normuon import build_optimizers 
import wandb

def train(model: TransformerLM, optimizers, train_data, batch_size, context_length, 
          max_iterations, device: torch.device, checkpoint_path, val_data = None, logger: ExperimentLogger = None,
          muon_lr_max=0.02, muon_lr_min=1e-3, adamw_lr_max=19e-4, adamw_lr_min=9e-5):
    expected_steps = 8000
    warmup_steps = int(.04*expected_steps)
    eval_interval = 1000
    checkpoint_interval = 5000
    model.train()

    muon_opt, adamw_opt = optimizers

    tokens_per_step = batch_size * context_length
    torch.cuda.synchronize()
    throughput_start = time.perf_counter()
    run_start = time.perf_counter()
    TIME_LIMIT = 45 * 60
    TIME_BUFFER = 110
    validation_counter = 0

    for iteration in range(1, max_iterations + 1):
        if time.perf_counter() - run_start >= TIME_LIMIT - TIME_BUFFER:
            model.eval()

            total_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_val, y_val in DataLoadingValidationSequential(
                    val_data, batch_size, context_length, device=device
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x_val)
                        loss = CrossEntropy(logits, y_val)

                    total_val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = total_val_loss / val_batches

            wandb.log({"val/loss": avg_val_loss}, step=iteration)

            break

        muon_lr = TrapezoidalLearningRateSchedule(
            alpha_max=muon_lr_max,
            alpha_min=muon_lr_max * 0.067,
            step=iteration,
            total_steps=expected_steps,warmup_steps=warmup_steps)
        adamw_lr = TrapezoidalLearningRateSchedule(
            alpha_max=adamw_lr_max,
            alpha_min=adamw_lr_max * 0.067,
            step=iteration,
            total_steps=expected_steps,warmup_steps=warmup_steps)

        if iteration % eval_interval == 0:
            validation_counter += 1
            model.eval()
            total_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_val, y_val in DataLoadingValidationSequential(val_data, batch_size, context_length, device=device):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x_val)
                        loss = CrossEntropy(logits, y_val)
                    total_val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                },
                step=iteration,)
            print(f"### Validation at {iteration}: Avg loss: {avg_val_loss:.4f} #####")
            model.train()

        x, y = DataLoading(train_data, batch_size, context_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = CrossEntropy(logits, y)

        muon_opt.zero_grad(set_to_none = True)
        adamw_opt.zero_grad(set_to_none = True)
        loss.backward()

        for group in muon_opt.param_groups:
            group["lr"] = muon_lr
        for group in adamw_opt.param_groups:
            group["lr"] = adamw_lr
        muon_opt.step()
        adamw_opt.step()

        if iteration % 50 == 0:
        # if True:

            torch.cuda.synchronize()

            elapsed = time.perf_counter() - throughput_start

            tokens_processed = 50 * tokens_per_step

            wandb.log(
            {
                "train/loss": loss.item(),
                "lr/muon": muon_lr,
                "lr/adamw": adamw_lr,
                # "optimizer/muon_momentum": momentum,
                "tokens_per_second": tokens_processed / elapsed,
            },
            step=iteration,
        )

            throughput_start = time.perf_counter()
            print(
                f"{iteration:5d} | "
                f"loss={loss.item():.4f} | "
                f"mu={muon_lr:.2e} | "
                # f"mom={momentum:.3f}"
            )

def main():
    checkpoint_path = "/checkpoints/owt_jul2_final_run.pth"
    d_model = 1024
    num_heads = 8
    d_ff = 4096
    num_layers = 16
    context_length = 512
    batch_size = 128
    # muon_lr = 0.029
    # adamw_lr = 28e-4
    muon_lr = 0.029
    adamw_lr = 0.0028
    max_iterations = 10000
    # max_iterations = 2
    weight_decay = 0.01
    theta = 10000.0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab_size = 32000

    config = {
        "d_model": d_model, "num_heads": num_heads, "d_ff": d_ff,
        "num_layers": num_layers, "context_length": context_length,
        "batch_size": batch_size, "muon_lr": muon_lr, "adamw_lr": adamw_lr,
        "weight_decay": weight_decay, "max_iterations": max_iterations, "theta": theta,
        "compile_mode": "reduce-overhead",
        "tf32": True,
        "bf16": True,
        "scheduler": "trapezoidal",
        "optimizer": "NorMuon+AdamW",
    }
    wandb.init(
        project="cs336-owt_July02_1",
        name="trapezoid_mu0.029_adam0.0028_final_run",
        config=config,
    )

    train_data = np.load("/data/owt_train.npy", mmap_mode="r")
    valid_data = np.load("/data/owt_valid.npy", mmap_mode="r")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # Recommended on PyTorch 2.x
    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta).to(device=device)
    model = torch.compile(model, mode="reduce-overhead")

    optimizers = build_optimizers(
        model, muon_lr=muon_lr, adamw_lr=adamw_lr,
        weight_decay=weight_decay, use_normuon=True)

    try:
        train(model, optimizers, train_data, batch_size, context_length, max_iterations,
              device, checkpoint_path, val_data=valid_data, logger=None,
              muon_lr_max=muon_lr, adamw_lr_max=adamw_lr)
    finally:
        wandb.finish()
