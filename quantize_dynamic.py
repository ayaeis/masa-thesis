import os
import time
import torch
import torch
torch.backends.quantized.engine = "qnnpack"
from moco.builder_dist import MASA

def make_dummy(B=1, T=256):
    rh = torch.randn(B, T, 21, 2)
    lh = torch.randn(B, T, 21, 2)
    body = torch.randn(B, T, 7, 2)
    mask = torch.ones(B, T, 49, 1)
    return {"rh": rh, "lh": lh, "body": body, "mask": mask}

def bench_ms(model, x, iters=50, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters

def sizeof_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def main():
    num_class = 100  # WLASL100
    model = MASA(skeleton_representation="graph-based", num_class=num_class, pretrain=False).cpu().eval()

    x = make_dummy(B=1, T=256)

    os.makedirs("quant_out", exist_ok=True)

    # Save FP32 state_dict
    fp32_path = "quant_out/masa_fp32.pth"
    torch.save(model.state_dict(), fp32_path)

    # Dynamic INT8 quantization (Linear layers)
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    int8_path = "quant_out/masa_int8_dynamic.pth"
    torch.save(qmodel.state_dict(), int8_path)

    # Quick sanity + benchmark
    fp32_ms = bench_ms(model, x)
    int8_ms = bench_ms(qmodel, x)

    print("Saved:", fp32_path, f"({sizeof_mb(fp32_path):.2f} MB)")
    print("Saved:", int8_path, f"({sizeof_mb(int8_path):.2f} MB)")
    print(f"FP32 avg latency: {fp32_ms:.2f} ms")
    print(f"INT8 avg latency: {int8_ms:.2f} ms")
    print(f"Speedup: {fp32_ms / max(int8_ms, 1e-9):.2f}x")

if __name__ == "__main__":
    main()