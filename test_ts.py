import torch
from palm_pytorch import PaLM

palm = PaLM(
    num_tokens=20000,
    dim=512,
    depth=12,
    heads=8,
    dim_head=64,
)

tokens = torch.randint(0, 20000, (1, 2048))
logits = palm(tokens)  # (1, 2048, 20000)

print(logits)
