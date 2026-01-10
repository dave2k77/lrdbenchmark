import torch
device = torch.device('cuda')
print("Creating small tensor...")
x = torch.ones(2, 2, device=device)
print("Matmul...")
y = x @ x
print(f"Result: {y}")
print("✅ Success")
