
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
