import torch
from moco.builder_dist import MASA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model in inference mode
model = MASA(
    skeleton_representation="graph-based",
    num_class=100,
    pretrain=False
)

model.to(device)
model.eval()

# Dummy input
B = 1
T = 32

rh = torch.randn(B, T, 21, 2).to(device)
lh = torch.randn(B, T, 21, 2).to(device)
body = torch.randn(B, T, 7, 2).to(device)
mask = torch.ones(B, 2 * T, 1, 1).to(device)

x = {
    "rh": rh,
    "lh": lh,
    "body": body,
    "mask": mask
}

with torch.no_grad():
    out = model(x)

print("Output shape:", out.shape)
