"""Negative-stain envelope as a differentiable target.

Envelope = map binarised at the contour that encloses the expected protein volume (mass x 1.21 A^3/Da),
lightly smoothed, cropped and resampled to a coarse grid. A model is compared to it by Pearson correlation
of a Gaussian-blurred token (one point per residue) density. Everything is torch so the same code scores
finished models (envfit) and steers diffusion (boltz_guided.py).

Convention: map-frame coords Y = X @ R + t (row vectors).
"""
import math
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

A3_PER_DA = 1.21


def read_map(path):
    with mrcfile.open(path, permissive=True) as m:
        d = np.asarray(m.data, dtype=np.float32)
        h = m.header
        vs = np.array([float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)])
        axes = (int(h.mapc), int(h.mapr), int(h.maps))
        start = np.array([int(h.nxstart), int(h.nystart), int(h.nzstart)])
        org = np.array([float(h.origin.x), float(h.origin.y), float(h.origin.z)])
    # data is (sections, rows, cols); reorder to array[x, y, z]
    d = np.transpose(d, (2, 1, 0))  # (cols, rows, sections)
    perm = [axes.index(i) for i in (1, 2, 3)]  # which array axis holds x, y, z
    d = np.transpose(d, perm)
    start = start[perm]
    if not np.any(org):
        org = start * vs
    return d, vs, org


class Envelope:
    def __init__(self, map_path, mass_da, resolution, voxel=None, margin=12.0, device="cpu"):
        d, vs, org = read_map(map_path)
        self.resolution = float(resolution)
        voxel = voxel or max(float(vs[0]), min(6.0, self.resolution / 4.0))
        target_vol = mass_da * A3_PER_DA
        # contour enclosing the expected volume
        n_vox = int(round(target_vol / float(np.prod(vs))))
        flat = np.sort(d.ravel())[::-1]
        n_vox = min(max(n_vox, 1), flat.size - 1)
        self.contour = float(flat[n_vox])
        mask = (d >= self.contour)
        idx = np.argwhere(mask)
        lo = np.maximum(idx.min(0) * vs - margin, 0)
        hi = np.minimum(idx.max(0) * vs + margin, (np.array(d.shape) - 1) * vs)
        # resample binary envelope onto coarse grid (trilinear on fine binary = partial volume)
        grid = [torch.arange(lo[i], hi[i] + 1e-6, voxel, dtype=torch.float32) for i in range(3)]
        t = torch.from_numpy(mask.astype(np.float32))[None, None]  # [1,1,X,Y,Z]
        gx, gy, gz = torch.meshgrid(*grid, indexing="ij")
        # grid_sample wants (z,y,x)-ordered last dim for input [N,C,D,H,W] with D=X here -> use (Z,Y,X) normalised
        norm = lambda g, n, s: 2 * g / ((n - 1) * s) - 1
        samp = torch.stack([norm(gz, d.shape[2], vs[2]), norm(gy, d.shape[1], vs[1]), norm(gx, d.shape[0], vs[0])], -1)
        env = F.grid_sample(t, samp[None], align_corners=True)[0, 0]
        env = gaussian_blur3d(env, 1.0)
        self.env = ((env - env.mean()) / env.std()).to(device)
        self.axes = [(g + float(org[i])).to(device) for i, g in enumerate(grid)]  # absolute coords (A)
        self.center = torch.tensor(idx.mean(0) * vs + org, dtype=torch.float32, device=device)
        self.sigma = max(0.225 * self.resolution, 3.0)
        self.device = device
        self.voxel = voxel
        self.env_volume = float(mask.sum() * np.prod(vs))

    def to(self, device):
        self.env = self.env.to(device)
        self.axes = [a.to(device) for a in self.axes]
        self.center = self.center.to(device)
        self.device = device
        return self

    def simulate(self, Y):
        """Y: [B,N,3] map-frame token coords -> [B,X,Y,Z] blurred density."""
        s2 = 2 * self.sigma ** 2
        g = [torch.exp(-(Y[..., i, None] - self.axes[i]) ** 2 / s2) for i in range(3)]  # [B,N,n_i]
        return torch.einsum("bni,bnj,bnk->bijk", g[0], g[1], g[2])

    def cc(self, Y):
        """Pearson CC of each model with the envelope: [B]."""
        S = self.simulate(Y).flatten(1)
        S = S - S.mean(1, keepdim=True)
        S = S / (S.norm(dim=1, keepdim=True) + 1e-8)
        E = self.env.flatten()
        E = E / E.norm()
        return S @ E


def gaussian_blur3d(v, sigma_vox):
    r = max(1, int(math.ceil(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma_vox ** 2)); k = k / k.sum()
    v = v[None, None]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + dim] = -1
        pad = [0] * 6; pad[2 * (2 - dim)] = r; pad[2 * (2 - dim) + 1] = r
        v = F.conv3d(F.pad(v, pad, mode="replicate"), k.view(shape))
    return v[0, 0]


# ---------------- rigid-body fitting ----------------

def axis_angle_to_R(w):
    """w: [...,3] -> [...,3,3] (Rodrigues)."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def random_rotations(n, device, generator=None):
    q = torch.randn(n, 4, generator=generator).to(device)
    q = q / q.norm(dim=1, keepdim=True)
    a, b, c, d = q.unbind(1)
    R = torch.stack([
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
        2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d], 1).view(n, 3, 3)
    return R


def kabsch(X, Y):
    """Best R,t with X@R+t ~ Y. X,Y: [B,N,3]."""
    mx, my = X.mean(1, keepdim=True), Y.mean(1, keepdim=True)
    H = (X - mx).transpose(1, 2) @ (Y - my)
    U, S, Vt = torch.linalg.svd(H.cpu())
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = (U @ D @ Vt).to(X.device)
    t = my - mx @ R
    return R, t


def refine_pose(env, X, R, t, iters=20, lr_rot=0.02, lr_tr=1.0):
    """Local rigid refinement maximising CC. X: [B,N,3]; R: [B,3,3]; t: [B,1,3]."""
    R, t = R.detach(), t.detach()
    with torch.enable_grad():
        w = torch.zeros(X.shape[0], 3, device=X.device, requires_grad=True)
        dt = torch.zeros_like(t, requires_grad=True)
        opt = torch.optim.Adam([{"params": [w], "lr": lr_rot}, {"params": [dt], "lr": lr_tr}])
        cen = X.mean(1, keepdim=True)
        for _ in range(iters):
            Rn = axis_angle_to_R(w) @ R
            Y = (X - cen) @ Rn + cen @ R + t + dt
            loss = -env.cc(Y).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        Rn = (axis_angle_to_R(w) @ R).detach()
        tn = (cen @ R + t + dt - cen @ Rn).detach()
    with torch.no_grad():
        c = env.cc(X @ Rn + tn)
    return Rn, tn, c


@torch.no_grad()
def global_fit(env, X, n_rot=300, chunk=16, top=4, refine_iters=40, seed=0):
    """Exhaustive-ish rigid fit of each model in X ([B,N,3]) to the envelope. Returns R [B,3,3], t [B,1,3], cc [B]."""
    g = torch.Generator().manual_seed(seed)
    Rs = torch.cat([torch.eye(3, device=X.device)[None], random_rotations(n_rot, X.device, g)])
    outR, outt, outc = [], [], []
    for b in range(X.shape[0]):
        x = X[b:b + 1] - X[b:b + 1].mean(1, keepdim=True)
        scores = []
        for i in range(0, len(Rs), chunk):
            Y = x @ Rs[i:i + chunk] + env.center
            scores.append(env.cc(Y))
        scores = torch.cat(scores)
        best = scores.topk(min(top, len(scores))).indices
        Rb = Rs[best]
        tb = (env.center - X[b:b + 1].mean(1, keepdim=True) @ Rb)
        Rr, tr, cr = refine_pose(env, X[b:b + 1].expand(len(best), -1, -1), Rb, tb, iters=refine_iters)
        k = int(cr.argmax())
        outR.append(Rr[k]); outt.append(tr[k]); outc.append(cr[k])
    return torch.stack(outR), torch.stack(outt), torch.stack(outc)
