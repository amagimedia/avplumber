# Corrected warp: precompute grid already in [-1, 1] normalized space and add
# normalized flow directly. The previous shim built the grid in pixel space
# and mixed units, causing flow to be scaled to ~0 -- the model degraded to a
# straight img0/img1 alpha blend with no motion warping.
import sys, os, torch, torch.nn as nn, torch.nn.functional as F, types
sys.path.insert(0, "/root/rife426")
sys.path.insert(0, "/root/rife426/RIFEv4.26_0921")

def _warp(tenInput, tenFlow):
    device, dtype = tenInput.device, tenInput.dtype
    b, c, h, w = tenInput.shape
    horiz = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1,1,1,w).expand(b,1,h,w)
    vert  = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1,1,h,1).expand(b,1,h,w)
    grid = torch.cat([horiz, vert], dim=1)
    tenFlow = torch.cat([
        tenFlow[:, 0:1] / ((w - 1.0) / 2.0),
        tenFlow[:, 1:2] / ((h - 1.0) / 2.0),
    ], dim=1)
    g = (grid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(tenInput, g, mode="bilinear", padding_mode="border", align_corners=True)

warplayer_mod = types.ModuleType("model.warplayer")
warplayer_mod.warp = _warp
sys.modules["model"] = types.ModuleType("model")
sys.modules["model.warplayer"] = warplayer_mod

os.makedirs("/root/rife426/train_log", exist_ok=True)
from train_log.IFNet_HDv3 import IFNet

model = IFNet().eval().cuda()
sd = torch.load("/root/rife426/RIFEv4.26_0921/flownet.pkl", map_location="cuda", weights_only=False)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model = model.half()

class RIFEExport(nn.Module):
    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet
    def forward(self, img0, img1, timestep):
        x = torch.cat((img0, img1), 1)
        scale_list = [16.0, 8.0, 4.0, 2.0, 1.0]
        _, _, merged = self.flownet(x, timestep, scale_list)
        return merged[-1]

wrap = RIFEExport(model).eval().cuda()

H, W = 1088, 1920
img0 = torch.rand(1, 3, H, W, device="cuda", dtype=torch.float16)
img1 = torch.rand(1, 3, H, W, device="cuda", dtype=torch.float16)
ts = torch.tensor([[[[0.5]]]], device="cuda", dtype=torch.float16)
with torch.no_grad():
    out = wrap(img0, img1, ts)
print("sanity forward ok:", out.shape, out.dtype)

out_path = "/root/rife426/rife_v4.26_fp16.onnx"
for p in [out_path, out_path + ".data"]:
    try: os.remove(p)
    except FileNotFoundError: pass

torch.onnx.export(
    wrap, (img0, img1, ts), out_path,
    input_names=["img0", "img1", "timestep"],
    output_names=["mid"],
    dynamic_axes={
        "img0": {0: "B"},
        "img1": {0: "B"},
        "timestep": {0: "B"},
        "mid":  {0: "B"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print("exported:", out_path)
