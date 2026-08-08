"""Check the freshly trained generator matches the dataset it will be used on."""
import sys, os
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

# Storage value comes from the carbon target now, not from price arbitrage, so
# the carbon channel and the net load that drives it are what reconstruction
# quality has to be judged on. Price spread is close to a constant here (one
# value on weekdays, zero at weekends) and barely enters the decision.
def _peak_net(s):
    load = s.active_load_mw.sum(axis=(1, 2))
    pv = s.pv_available_mw.sum(axis=(1, 2))
    return float((load - pv).max())


print(f"\n{'scenario':<18}{'channel':<16}{'real':>9}{'recon':>9}{'error':>9}")
print("-" * 62)
for i, s in enumerate(rebuilt):
    real = pool.scenarios[i]
    rows = [
        ("carbon mean", float(real.grid_carbon_t_per_mwh.mean()),
         float(s.grid_carbon_t_per_mwh.mean())),
        ("carbon swing", float(np.ptp(real.grid_carbon_t_per_mwh)),
         float(np.ptp(s.grid_carbon_t_per_mwh))),
        ("peak net load", _peak_net(real), _peak_net(s)),
        ("price spread", float(np.ptp(real.grid_price_per_mwh)),
         float(np.ptp(s.grid_price_per_mwh))),
    ]
    for j, (name, a, b) in enumerate(rows):
        err = f"{100 * (b - a) / a:+.0f}%" if a else "n/a"
        label = real.name if j == 0 else ""
        print(f"{label:<18}{name:<16}{a:>9.3f}{b:>9.3f}{err:>9}")
    print()


# The four examples above are useful diagnostics, but acceptance must use the
# complete split. Decode posterior means in batches and summarize absolute
# relative errors. Zero price spreads are checked separately.
decoded_batches = []
with torch.no_grad():
    for start in range(0, len(pool.scenarios), 32):
        stop = min(start + 32, len(pool.scenarios))
        batch_t = torch.as_tensor(traj[start:stop], dtype=torch.float32, device=device)
        batch_c = torch.as_tensor(ctx[start:stop], dtype=torch.float32, device=device)
        batch_latent, _ = gen.encode(batch_t, batch_c)
        decoded_batches.append(gen.decode(batch_latent, batch_c).cpu().numpy())

decoded_all = np.concatenate(decoded_batches, axis=0)
rebuilt_all = codec.decode_batch(decoded_all, ctx, name_prefix="audit")


def _metrics(s):
    return np.asarray(
        [
            float(s.grid_carbon_t_per_mwh.mean()),
            float(np.ptp(s.grid_carbon_t_per_mwh)),
            _peak_net(s),
            float(np.ptp(s.grid_price_per_mwh)),
        ],
        dtype=float,
    )


labels = ("carbon mean", "carbon swing", "peak net load", "price spread")
real_values = np.stack([_metrics(s) for s in pool.scenarios])
recon_values = np.stack([_metrics(s) for s in rebuilt_all])
print(f"all-scenario posterior-mean audit ({len(pool.scenarios)} scenarios)")
print(f"{'channel':<16}{'median |err|':>15}{'p90 |err|':>13}{'max |err|':>13}")
print("-" * 57)
for column, label in enumerate(labels):
    nonzero = np.abs(real_values[:, column]) > 1.0e-9
    relative = 100.0 * np.abs(
        (recon_values[nonzero, column] - real_values[nonzero, column])
        / real_values[nonzero, column]
    )
    print(
        f"{label:<16}{np.median(relative):>14.1f}%"
        f"{np.quantile(relative, 0.90):>12.1f}%{relative.max():>12.1f}%"
    )

flat_price = np.abs(real_values[:, 3]) <= 1.0e-9
if flat_price.any():
    reconstructed_flat_spread = recon_values[flat_price, 3]
    print(
        f"flat-price cases : {int(flat_price.sum())}; "
        f"maximum reconstructed spread {reconstructed_flat_spread.max():.6f}"
    )
