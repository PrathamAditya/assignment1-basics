import modal
from cs336_basics.training_together import main as train_main

app = modal.App("cs336-assignment1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("cs336_basics")
)

data_volume = modal.Volume.from_name("cs336-data")
checkpoint_volume = modal.Volume.from_name("checkpoints")
logs_volume = modal.Volume.from_name("experiment-logs")

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 45,
    secrets=[
        modal.Secret.from_name("wandb-secret")
    ],
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/logs": logs_volume,
    },
)
def train():
    train_main()


@app.local_entrypoint()
def main():
    print("Calling remote train.")
    train.remote()