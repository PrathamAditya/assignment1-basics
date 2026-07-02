import time
import json
import os
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ExperimentLogger:
    def __init__(self, experiment_name: str, config: dict, log_dir: str = "experiment_logs", use_wandb: bool = True):
        """
        Initializes the logging infrastructure.
        
        Args:
            experiment_name: Name of the current ablation or run (e.g., "baseline_17M", "no_layer_norm").
            config: Hyperparameters for this run (lr, batch size, etc.).
            log_dir: Local directory to save the JSON log files.
            use_wandb: Whether to sync with Weights & Biases.
        """
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Filepath for the local deliverable log
        self.log_filepath = self.log_dir / f"{experiment_name}_log.json"
        
        # Tracking variables
        self.start_time = time.time()
        self.history = []
        
        # Initialize Weights & Biases if requested and available
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="transformer-tinystories",
                name=self.experiment_name,
                config=self.config
            )
            
        print(f"[*] Started experiment: {self.experiment_name}")
        self._save_local_log()

    def log_step(self, step: int, train_loss: float = None, val_loss: float = None, lr: float = None, custom_metrics: dict = None):
        """
        Logs metrics for a specific gradient step.
        """
        current_time = time.time()
        wall_clock_time = current_time - self.start_time
        
        metrics = {
            "step": step,
            "wall_clock_time_seconds": wall_clock_time,
        }
        if train_loss is not None:
            metrics["train_loss"] = train_loss
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if custom_metrics:
            metrics.update(custom_metrics)
        if lr is not None:
            metrics["learning_rate"] = lr
            
        # 1. Update local history and save
        self.history.append(metrics)
        self._save_local_log()
        
        # 2. Sync to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def _save_local_log(self):
        """Saves the experiment record to a local JSON file."""
        log_data = {
            "experiment_name": self.experiment_name,
            "hyperparameters": self.config,
            "total_runtime_seconds": time.time() - self.start_time,
            "history": self.history
        }
        with open(self.log_filepath, "w") as f:
            json.dump(log_data, f, indent=4)

    def finish(self):
        """Closes the logger when training is complete."""
        self._save_local_log()

        final_train_loss = None
        final_val_loss = None

        for record in reversed(self.history):
            if final_train_loss is None and "train_loss" in record:
                final_train_loss = record["train_loss"]

            if final_val_loss is None and "val_loss" in record:
                final_val_loss = record["val_loss"]

            if final_train_loss is not None and final_val_loss is not None:
                break

        summary = {
            "experiment_name": self.experiment_name,
            "runtime_seconds": time.time() - self.start_time,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "total_steps": self.history[-1]["step"] if self.history else 0,
        }

        summary_file = self.log_dir / "summary.jsonl"

        with open(summary_file, "a") as f:
            f.write(json.dumps(summary) + "\n")

        if self.use_wandb:
            wandb.finish()

        print(f"[*] Finished experiment: {self.experiment_name}")
        print(summary)