#!/usr/bin/env python3
"""Minimal ncu profiling target: runs exactly 2 matmuls to profile."""
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
H, I = 2048, 768
C, M = 32, 16
x = torch.randn(1, H, dtype=torch.bfloat16, device=device)

# Dense
W = torch.randn(I, H, dtype=torch.bfloat16, device=device)
for _ in range(5):
    torch.matmul(x, W.t())
torch.cuda.synchronize()

# SA cached
Wc = torch.randn(C, H, dtype=torch.bfloat16, device=device)
for _ in range(5):
    torch.matmul(x, Wc.t())
torch.cuda.synchronize()

# SA missed
Wm = torch.randn(M, H, dtype=torch.bfloat16, device=device)
for _ in range(5):
    torch.matmul(x, Wm.t())
torch.cuda.synchronize()

print("ncu target done")
