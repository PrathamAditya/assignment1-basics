import torch

def CrossEntropy(logits: torch.Tensor, target_token_ids: torch.Tensor):
    temp_tensor = torch.max(logits, dim = -1)
    temp_tensor = temp_tensor.values.unsqueeze(-1)
    logits = logits - temp_tensor
    exp_logits = torch.exp(logits) # exp last dim
    sum_exp = torch.sum(exp_logits, dim=-1)
    log_logits = torch.log(sum_exp)
    o_result = torch.gather(logits, dim=-1, index=target_token_ids.unsqueeze(-1))
    loss = log_logits - o_result.unsqueeze(-1)
    return loss.mean()
