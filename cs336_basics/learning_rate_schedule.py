
import math

def LearningRateSchedule(alpha_max: float, alpha_min: float, t: int, T_w: int, T_c: int):
    # Warm-up
    if t < T_w:
        return (t/T_w)*alpha_max
    # Cosine annealing
    elif T_w <= t <= T_c:
        return alpha_min + 1/2*(1 + math.cos(((t-T_w)/(T_c-T_w))*math.pi))*(alpha_max-alpha_min)
    # Post-annealing
    else:
        return alpha_min

def TrapezoidalLearningRateSchedule(
    alpha_max,
    alpha_min,
    step,
    total_steps,
    warmup_steps=200,
    plateau_fraction=0.20,
):
    decay_steps = total_steps - warmup_steps
    plateau_steps = int(decay_steps * plateau_fraction)
    plateau_end = warmup_steps + plateau_steps

    if step <= warmup_steps:
        return alpha_max * step / warmup_steps

    if step <= plateau_end:
        return alpha_max
    
    if step >= total_steps:
        return alpha_min

    progress = (step - plateau_end) / (total_steps - plateau_end)
    return alpha_max + progress * (alpha_min - alpha_max)