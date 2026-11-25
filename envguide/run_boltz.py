"""Run Boltz-2 on one target, optionally steering diffusion toward the NS envelope (option 2).

  python envguide/run_boltz.py EMD-6339 --mode unguided --samples 5
  python envguide/run_boltz.py EMD-6339 --mode guided   --samples 5 --scale 1.0

Guidance: at each denoising step the x0 prediction (one point per token) is rigidly registered to the envelope
(warm start by Kabsch onto the previous step's registered x0, local CC refinement, periodic global search),
then nudged to raise envelope CC: per-chain rigid moves plus a smaller per-token term, with backtracking so a
step is only kept if CC goes up. Nothing in Boltz is retrained; the network's denoiser is unchanged.

Outputs: /Volumes/boltzcache/envguide_runs/<mode>[_tag]/<EMD>/ (Boltz output tree) and guide_log.json (CC per step and sample).
"""
import argparse, csv, json, os, sys, time
# 16 GB M1: cap the MPS allocator so an oversized job fails instead of swapping the (nearly full) system disk
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from envelope import Envelope, global_fit, refine_pose, kabsch, axis_angle_to_R  # noqa: E402


class EnvelopeGuide:
    def __init__(self, env, n_steps, scale=1.0, start_frac=0.05, global_every=25, gd_steps=2,
                 token_weight=0.1, log=None):
        self.env, self.n_steps, self.scale = env, n_steps, scale
        self.start = int(start_frac * n_steps)
        self.global_every, self.gd_steps, self.token_weight = global_every, gd_steps, token_weight
        self.step = -1
        self.R = self.t = self.Yprev = None
        self.log = log if log is not None else []

    def tokens(self, x, feats):
        rep = feats["token_to_rep_atom"][0].float()  # [T,A]
        mask = feats["token_pad_mask"][0].bool()
        rep = rep[mask]
        return torch.einsum("na,bad->bnd", rep, x.float()), rep, mask

    def register(self, X):
        """Pose of current x0 tokens X [B,N,3] in the map frame."""
        if self.Yprev is None or (self.step - self.start) % self.global_every == 0:
            Rg, tg, cg = global_fit(self.env, X, n_rot=200, refine_iters=25, seed=self.step)
        if self.Yprev is not None:
            Rw, tw = kabsch(X, self.Yprev)
            Rw, tw, cw = refine_pose(self.env, X, Rw, tw, iters=8)
            if (self.step - self.start) % self.global_every == 0:
                take = (cg > cw)[:, None, None]
                Rw, tw, cw = torch.where(take, Rg, Rw), torch.where(take, tg, tw), torch.maximum(cg, cw)
            return Rw, tw, cw
        return Rg, tg, cg

    def __call__(self, x0, feats, sigma):
        self.step += 1
        if self.step < self.start or self.step >= self.n_steps - 1:
            return x0
        dev = x0.device
        X, rep, mask = self.tokens(x0, feats)  # X [B,N,3]
        R, t, c0 = self.register(X)
        asym = feats["asym_id"][0][mask].long()
        chains = torch.unique(asym)
        onehot = (asym[:, None] == chains[None]).float()  # [N,C]
        cnt = onehot.sum(0)
        disp = torch.zeros_like(X)
        cur = c0
        for _ in range(self.gd_steps):
            with torch.enable_grad():
                w = torch.zeros(X.shape[0], len(chains), 3, device=dev, requires_grad=True)
                tt = torch.zeros_like(w, requires_grad=True)
                d_tok = torch.zeros_like(X, requires_grad=True)
                Xc = X + disp
                cen = (onehot.T @ Xc) / cnt[None, :, None]  # [B,C,3]
                Rc = axis_angle_to_R(w)  # [B,C,3,3]
                rel = Xc - cen[:, asym]
                Xm = torch.einsum("bnd,bnde->bne", rel, Rc[:, asym]) + cen[:, asym] + tt[:, asym] + d_tok
                cc = self.env.cc(Xm @ R + t)
                cc.sum().backward()
            # unit-norm moves: chains up to `scale` A translation, rotation giving ~scale A at chain radius
            rad = (((Xc - cen[:, asym]) ** 2).sum(-1).sqrt() @ onehot / cnt).clamp_min(5.0)  # [B,C]
            gt, gw, gd = tt.grad, w.grad, d_tok.grad
            step_t = gt / (gt.norm(dim=-1, keepdim=True) + 1e-12)
            step_w = gw / (gw.norm(dim=-1, keepdim=True) + 1e-12) / rad[..., None]
            step_d = gd / (gd.norm(dim=-1, keepdim=True).amax(1, keepdim=True) + 1e-12)
            accepted = torch.zeros(X.shape[0], dtype=torch.bool, device=dev)
            new_disp = disp.clone()
            for f in (1.0, 0.5, 0.25):
                a = self.scale * f
                with torch.no_grad():
                    Rc = axis_angle_to_R(a * step_w)
                    rel = Xc - cen[:, asym]
                    Xn = torch.einsum("bnd,bnde->bne", rel, Rc[:, asym]) + cen[:, asym] + a * step_t[:, asym] \
                        + a * self.token_weight * step_d
                    cn = self.env.cc(Xn @ R + t)
                ok = (cn > cur) & ~accepted
                new_disp[ok] = (Xn - X)[ok]
                cur = torch.where(ok, cn, cur)
                accepted |= ok
                if accepted.all():
                    break
            disp = new_disp
        self.Yprev = ((X + disp) @ R + t).detach()
        self.log.append(dict(step=self.step, sigma=float(sigma), cc_before=c0.tolist(), cc_after=cur.tolist()))
        # token displacement -> atoms (each atom moves with its token's representative atom)
        a2t = feats["atom_to_token"][0].float()[:, mask]  # [A,N]
        return x0 + torch.einsum("an,bnd->bad", a2t, disp).to(x0.dtype)


def install(guide_factory):
    import boltz.main
    from boltz.model.modules import diffusionv2
    # Lightning predicts under torch.inference_mode, where enable_grad() has no effect and the guide's
    # backward pass fails; no_grad still lets enable_grad() turn autograd back on.
    trainer_cls = boltz.main.Trainer
    boltz.main.Trainer = lambda *a, **kw: trainer_cls(*a, inference_mode=False, **kw)
    orig_sample = diffusionv2.AtomDiffusion.sample
    orig_fwd = diffusionv2.AtomDiffusion.preconditioned_network_forward

    def sample(self, atom_mask, num_sampling_steps=None, multiplicity=1, max_parallel_samples=None,
               steering_args=None, **kw):
        n = num_sampling_steps or self.num_sampling_steps
        self._env_guide = guide_factory(n)
        self._env_feats = kw["feats"]
        try:
            return orig_sample(self, atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity,
                               max_parallel_samples=None, steering_args=steering_args, **kw)
        finally:
            self._env_guide = None

    def fwd(self, noised, sigma, network_condition_kwargs):
        x0 = orig_fwd(self, noised, sigma, network_condition_kwargs)
        g = getattr(self, "_env_guide", None)
        if g is not None:
            x0 = g(x0, self._env_feats, sigma)
        return x0

    diffusionv2.AtomDiffusion.sample = sample
    diffusionv2.AtomDiffusion.preconditioned_network_forward = fwd


MSA_DIR = os.path.join(HERE, "msa")
# APFS sparse image on the KINGSTON drive (external/boltzcache.sparsebundle); FAT32 is too slow for the CCD files.
# Mount with: hdiutil attach external/boltzcache.sparsebundle
BOLTZ_CACHE = os.environ.get("BOLTZ_CACHE", "/Volumes/boltzcache")  # override on Colab/Linux
# Boltz outputs live there too: on FAT32 macOS adds ._*.json sidecars that Boltz tries to parse as records.
RUNS = os.environ.get("ENVGUIDE_RUNS") or os.path.join(BOLTZ_CACHE, "envguide_runs")  # ENVGUIDE_RUNS: other run folder


def msa_input(emd):
    """Reuse MSAs from an earlier run (so every mode sees the same MSA); else query the ColabFold server."""
    src = os.path.join(HERE, "inputs", f"{emd}.yaml")
    lines = open(src).read().splitlines()
    n_ent = sum(1 for l in lines if l.strip() == "- protein:")
    msas = [os.path.join(MSA_DIR, f"{emd}_{i}.csv") for i in range(n_ent)]
    if not all(os.path.exists(m) for m in msas):
        return src, True
    out, i = [], 0
    for l in lines:
        out.append(l)
        if l.strip().startswith("sequence:"):
            out.append(f"      msa: {msas[i]}"); i += 1
    dst = os.path.join(MSA_DIR, f"{emd}.yaml")
    open(dst, "w").write("\n".join(out) + "\n")
    return dst, False


def save_msas(emd, out):
    import glob, shutil
    os.makedirs(MSA_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(out, "boltz_results_*", "msa", f"{emd}_*.csv")):
        dst = os.path.join(MSA_DIR, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst)


def target_row(emd):
    for r in csv.DictReader(open(os.path.join(HERE, "targets.csv"))):
        if r["emdb"] == emd:
            return r
    raise SystemExit(f"{emd} not in targets.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emd")
    ap.add_argument("--mode", choices=["unguided", "guided"], required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--token-weight", type=float, default=0.1)
    ap.add_argument("--start-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="mps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--msa-depth", type=int, default=1024, help="MSA rows used per pass (memory)")
    a = ap.parse_args()

    row = target_row(a.emd)
    out = os.path.join(RUNS, a.mode + (f"_{a.tag}" if a.tag else ""), a.emd)
    os.makedirs(out, exist_ok=True)
    log = []
    if a.mode == "guided":
        dev = {"mps": "mps", "gpu": "cuda"}.get(a.accelerator, "cpu")
        env = Envelope(os.path.join(ROOT, "maps_unzipped", f"{a.emd}.map"), float(row["mass_da"]),
                       float(row["resolution"]), device=dev)
        install(lambda n: EnvelopeGuide(env, n, scale=a.scale, start_frac=a.start_frac,
                                        token_weight=a.token_weight, log=log))
    from boltz.main import predict
    data, use_server = msa_input(a.emd)
    t0 = time.time()
    predict.callback(
        data=data, out_dir=out, use_msa_server=use_server,
        cache=BOLTZ_CACHE, accelerator=a.accelerator,
        diffusion_samples=a.samples, sampling_steps=a.steps, seed=a.seed,
        override=False, output_format="mmcif", num_workers=0, no_kernels=True,
        **{k: v for k, v in dict(checkpoint=None, affinity_checkpoint=None, devices=1, recycling_steps=a.recycling,
                                 sampling_steps_affinity=200, diffusion_samples_affinity=3,
                                 max_parallel_samples=None, step_scale=None, write_full_pae=False,
                                 write_full_pde=False, msa_server_url="https://api.colabfold.com",
                                 msa_pairing_strategy="greedy", msa_server_username=None,
                                 msa_server_password=None, api_key_header=None, api_key_value=None,
                                 use_potentials=False, model="boltz2", method=None,
                                 affinity_mw_correction=False, preprocessing_threads=1, max_msa_seqs=8192,
                                 subsample_msa=True, num_subsampled_msa=a.msa_depth, write_embeddings=False).items()})
    save_msas(a.emd, out)
    json.dump(dict(args=vars(a), seconds=round(time.time() - t0, 1), log=log),
              open(os.path.join(out, "guide_log.json"), "w"))
    print("done", a.emd, a.mode, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
