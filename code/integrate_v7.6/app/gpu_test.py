import torch

print("=== CUDA / GPU 診斷 ===")
print(f"torch version     : {torch.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")
print(f"CUDA device count : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    mem_gb = p.total_memory / 1024**3
    print(f"\n  CUDA Device {i}: {p.name}")
    print(f"    VRAM total  : {mem_gb:.1f} GB")
    print(f"    Compute Cap : {p.major}.{p.minor}")

if torch.cuda.is_available():
    print("\n--- 壓力測試（確認 GPU 真的在跑）---")
    import time
    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"✅ 10 次 4096x4096 矩陣乘法完成，耗時 {elapsed*1000:.1f}ms")
    print(f"   此時請觀察 Task Manager > GPU 1 (NVIDIA) 的 3D 使用率是否上升")

    used_mb = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   NVIDIA VRAM 使用量: {used_mb:.1f} MB")
else:
    print("\n❌ CUDA 不可用！所有 AI 推論都在跑 CPU，這就是超慢的原因。")
    print("   請確認：")
    print("   1. NVIDIA 驅動已安裝（nvidia-smi 有回應）")
    print("   2. PyTorch 版本與 CUDA 版本匹配（pip install torch --index-url https://download.pytorch.org/whl/cu121）")
