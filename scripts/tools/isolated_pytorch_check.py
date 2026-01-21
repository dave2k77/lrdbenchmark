import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU 0: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    
    device = torch.device('cuda')
    try:
        x = torch.randn(10, 10, device=device)
        y = torch.matmul(x, x.T)
        print("✅ GPU computation passed")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
