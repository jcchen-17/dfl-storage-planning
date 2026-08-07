"""Check the freshly trained generator matches the dataset it will be used on."""
import sys, os, json
sys.path.insert(0, "src")
os.environ.setdefault("STORAGE_DFL_SOLVER_WORKER", "1")
import torch
from storage_dfl.config import load_config
from storage_dfl.dfl import resolve_device
from storage_dfl.stages import (
    ArtifactPaths, _experiment_data, _load_codec, load_generator,
)

cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/generator_compare_k1t2.yaml"
c = load_config(cfg)
paths = ArtifactPaths(c.output_dir)
feeder, pool = _experiment_data(c, c.data.validation_split)
data_dim = int(pool.scenarios[0].context.size)
print(f"dataset            : {c.data.dataset_path.name}")
print(f"  scenarios        : {len(pool.scenarios)}")
print(f"  context dim      : {data_dim}")

codec = _load_codec(paths, feeder)
print(f"normalization.json : context_dim {codec.context_dim}"
      f"  {'OK' if codec.context_dim == data_dim else 'MISMATCH'}")

traj, ctx = codec.encode_pool(pool)
print(f"encode_pool        : trajectories {traj.shape}, contexts {ctx.shape}  OK")

device = resolve_device(c.dfl.device)
gen = load_generator(paths, c.generator.kind, device)
with torch.no_grad():
    t = torch.as_tensor(traj[:4], dtype=torch.float32, device=device)
    x = torch.as_tensor(ctx[:4], dtype=torch.float32, device=device)
    latent, _ = gen.encode(t, x)
    decoded = gen.decode(latent, x).cpu().numpy()
print(f"{c.generator.kind}.pt round trip : latent {tuple(latent.shape)}"
      f" -> decoded {decoded.shape}  OK")

rebuilt = codec.decode_batch(decoded, ctx[:4], name_prefix="probe")
import numpy as np
for i, s in enumerate(rebuilt):
    real = pool.scenarios[i]
    print(f"  {real.name:<18} price spread real {float(np.ptp(real.grid_price_per_mwh)):7.2f}"
          f"  reconstructed {float(np.ptp(s.grid_price_per_mwh)):7.2f}"
          f"   | peak load real {float(real.active_load_mw.sum(axis=(1,2)).max()):5.3f}"
          f"  recon {float(s.active_load_mw.sum(axis=(1,2)).max()):5.3f}")
