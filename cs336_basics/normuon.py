import math
import torch


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate orthogonalization of G via quintic Newton-Schulz iteration
    (Keller Jordan's coefficients, tuned for fast convergence to the
    nearest semi-orthogonal matrix in the SVD sense: G -> U V^T).

    G: (out_features, in_features)
    """
    assert G.ndim == 2, f"newton_schulz_orthogonalize expects a 2D tensor, got shape {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    X = X / (X.norm() + eps)  # normalize spectral norm roughly to <= 1

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer: momentum SGD + Newton-Schulz orthogonalization.
    Only works on 2D parameters (weight matrices). Use AdamW separately
    for embeddings, output head, norms, and biases.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, (
                    f"Muon only supports 2D parameters, got shape {g.shape}. "
                    f"Make sure embeddings/biases/norms are routed to a different optimizer."
                )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = newton_schulz_orthogonalize(update, steps=ns_steps)

                # RMS-matching scale so update magnitude is comparable across
                # rectangular matrix shapes (standard Muon trick).
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                update = update * scale

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss


class NorMuon(torch.optim.Optimizer):
    """
    NorMuon: Muon + per-row (per-output-neuron) second-moment normalization
    of the momentum buffer, applied BEFORE Newton-Schulz orthogonalization.
    This is like Adam's variance adaptation but row-wise instead of
    element-wise, so it composes cleanly with orthogonalization.

    Only works on 2D parameters (weight matrices).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 beta2: float = 0.95, eps: float = 1e-8):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps, weight_decay=weight_decay,
                         beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, (
                    f"NorMuon only supports 2D parameters, got shape {g.shape}. "
                    f"Make sure embeddings/biases/norms are routed to a different optimizer."
                )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["row_var"] = torch.zeros(g.size(0), device=g.device, dtype=torch.float32)
                    state["step"] = 0

                state["step"] += 1
                t = state["step"]

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf

                # --- row-wise (per-output-neuron) second moment normalization ---
                row_var = state["row_var"]
                row_sq_mean = update.pow(2).mean(dim=1).to(torch.float32)  # (out_features,)
                row_var.mul_(beta2).add_(row_sq_mean, alpha=1 - beta2)

                bias_corr = 1 - beta2 ** t
                row_rms = (row_var / bias_corr).sqrt().add_(eps)  # (out_features,)

                update = update / row_rms.unsqueeze(1).to(update.dtype)
                # ------------------------------------------------------------------

                update = newton_schulz_orthogonalize(update, steps=ns_steps)

                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                update = update * scale

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss


def build_optimizers(model: torch.nn.Module, muon_lr: float = 0.02, adamw_lr: float = 3e-4,
                      weight_decay: float = 0.01, use_normuon: bool = True,
                      muon_momentum: float = 0.95, adamw_betas=(0.9, 0.95)):
    """
    Splits model parameters into:
      - muon_params: 2D weight matrices (attention/MLP linears) -> Muon/NorMuon
      - adamw_params: everything else (embeddings, lm_head, norms, biases) -> AdamW

    Returns a list of optimizers; call .step() / .zero_grad() on each.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name.lower() and "lm_head" not in name.lower():
            muon_params.append(p)
        else:
            adamw_params.append(p)

    MuonCls = NorMuon if use_normuon else Muon
    muon_opt = MuonCls(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=weight_decay)

    return [muon_opt, adamw_opt]