import numpy as np, scipy.fft as sf
from scipy.optimize import brentq

G = 8
ZONES = 15                       # zone = u+v of the 8x8 block DCT
LOG2 = np.log(2)

def load_yuv(path, H, W, n):
    raw = np.fromfile(path, dtype=np.uint8, count=int(n * H * W * 1.5)).reshape(n, -1)
    Y = raw[:, :H * W].reshape(n, H, W).astype(np.float32)
    U = raw[:, H * W:H * W * 5 // 4].reshape(n, H // 2, W // 2).astype(np.float32)
    V = raw[:, H * W * 5 // 4:].reshape(n, H // 2, W // 2).astype(np.float32)
    return [Y, U, V]

def blocks(x):                   # (g,h,w) -> (g,nby,nbx,8,8)
    g, h, w = x.shape
    return x.reshape(g, h // 8, 8, w // 8, 8).transpose(0, 1, 3, 2, 4)

def unblocks(b):                 # (nby,nbx,8,8) -> (h,w)
    nby, nbx = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(nby * 8, nbx * 8)

def fwd(x):                      # (G,h,w) -> coefficients (G,nby,nbx,8,8)
    c = sf.dctn(blocks(x), axes=(3, 4), norm="ortho")
    return sf.dct(c, axis=0, norm="ortho")

def last_frame(c):               # coefficients -> last frame pixels
    x = sf.idct(c, axis=0, norm="ortho")[-1]
    return unblocks(sf.idctn(x, axes=(2, 3), norm="ortho"))

def si_plane(frame):             # temporal-DC prediction from a (reconstructed) frame
    d = sf.dctn(blocks(frame[None])[0], axes=(2, 3), norm="ortho") * np.sqrt(G)
    z = np.zeros((G,) + d.shape, np.float32); z[0] = d
    return z

class Layout:
    """chunk map: (plane, t, zone u+v, region)."""
    def __init__(self, shapes, R=(4, 2, 2)):
        self.cid, self.off, self.nch, off = [], [], [], 0
        uv = (np.arange(8)[:, None] + np.arange(8)[None, :])
        for (h, w), r in zip(shapes, R):
            nby, nbx = h // 8, w // 8
            t = np.arange(G)[:, None, None, None, None]
            reg = ((np.arange(nby) * r // nby)[:, None] * r + (np.arange(nbx) * r // nbx)[None, :])[None, :, :, None, None]
            cid = ((t * ZONES + uv[None, None, None]) * r * r + reg) + off
            self.cid.append(cid.astype(np.int32).ravel()); self.off.append(off)
            off += G * ZONES * r * r
        self.sizes = [c.size for c in self.cid]
        self.Rp = list(R); self.plane_off = list(self.off)
        self.cid = np.concatenate(self.cid); self.nch = off
        self.n = np.bincount(self.cid, minlength=off).astype(np.float64); self.k = self.cid.size
        t0 = np.zeros(self.k, bool); p = 0
        for (h, w), s in zip(shapes, self.sizes):
            t0[p:p + (h // 8) * (w // 8) * 64] = True; p += s
        self.t0 = t0                                  # coefficients of temporal index 0
        self.t0_chunk = np.zeros(off, bool); self.t0_chunk[np.unique(self.cid[t0])] = True

W2 = sf.idct(np.eye(G), axis=0, norm="ortho")[G - 1, :] ** 2   # squared last-frame weights of the temporal coefficients

def si_error(L, D):
    """Error variance of the temporal-DC side information z built from the previous reconstructed
    last frame: Var = G * sum_t w_t^2 D_t(zone,region)  (temporal mixing, corrects the per-chunk recursion)."""
    out = np.zeros(L.nch)
    for r, off in zip(L.Rp, L.plane_off):
        per_t = ZONES * r * r
        Dt = D[off: off + G * per_t].reshape(G, per_t)
        out[off: off + per_t] = G * (W2[:, None] * Dt).sum(axis=0)
    return out

def q_db(x, step=1.0):
    return 10 ** (np.round(10 * np.log10(np.maximum(x, 1e-6)) / step) * step / 10)

def gop_stats(L, c, z0):
    """encoder-side statistics (open loop, from source frames only), quantised as metadata."""
    nj = L.n
    lam = np.bincount(L.cid, c.astype(np.float64) ** 2, L.nch) / nj
    if z0 is None:
        a = np.zeros(L.nch)
    else:
        num = np.bincount(L.cid, (c * z0).astype(np.float64), L.nch)
        den = np.bincount(L.cid, z0.astype(np.float64) ** 2, L.nch) + 1e-9
        a = np.clip(num / den, 0, 0.999) * L.t0_chunk
    res = c - a[L.cid].astype(np.float32) * (z0 if z0 is not None else 0)
    om = np.bincount(L.cid, res.astype(np.float64) ** 2, L.nch) / nj
    lam_q, om_q, a_q = q_db(lam), q_db(np.minimum(om, lam)), np.round(a * 32) / 32
    bits_meta = L.nch * 12 + L.t0_chunk.sum() * 5
    return lam_q, om_q, a_q, bits_meta

def floor_design(L, lam, om, a, Bc, c):
    nj = L.n
    f = lambda th: np.sum(nj * 0.5 * np.log2(np.maximum((om + a * a * th) / th, 1.0))) - Bc
    th = brentq(f, 1e-6, lam.max(), xtol=1e-12)
    DF = np.sum(nj * np.minimum(th, lam)) / L.k
    Fch = lam > th; inF = Fch[L.cid]
    cf = c[inF].astype(np.float64); lamF = lam[L.cid[inF]]
    def quant(delta):
        q = np.sign(cf) * np.floor(np.abs(cf) / delta + 1 / 3)
        return q
    def mse(delta):
        q = quant(delta)
        return (np.sum((cf - q * delta) ** 2) + np.sum(c[~inF].astype(np.float64) ** 2)) / L.k - DF
    lo, hi = 1e-3, 1e3
    if mse(lo) > 0: delta = lo
    else:
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            if mse(mid) > 0: hi = mid
            else: lo = mid
        delta = lo
    q = quant(delta); cF = np.zeros(L.k, np.float32); cF[inF] = (q * delta).astype(np.float32)
    # operational rate: Laplacian model with the transmitted variance (decoder-known)
    b = np.sqrt(lamF / 2) / delta
    m = np.abs(q)
    p0 = 1 - np.exp(-(2 / 3) / b)
    pm = 0.5 * (np.exp(-(m - 1 / 3) / b) - np.exp(-(m + 2 / 3) / b))
    p = np.where(m == 0, p0, np.maximum(pm, 1e-12))
    bits_lap = float(np.sum(-np.log2(np.maximum(p, 1e-12))))
    Rintra = float(np.sum(nj * 0.5 * np.log2(np.maximum(lam / th, 1.0))))
    return dict(theta=th, DF=DF, Fch=Fch, inF=inF, cF=cF, delta=delta, bits_lap=bits_lap, Rintra=Rintra,
                nF_coef=int(inF.sum()), mse_floor=(np.sum((c.astype(np.float64) - cF) ** 2) / L.k))

def alloc_chunks(kappa, vth, w, nu, P, M):
    def sel(xi):
        r = np.sqrt(xi * nu * vth)
        s = np.where(kappa > r, (kappa - r) ** 2 / kappa, 0.0)
        o = np.argsort(-s); cum = np.cumsum(w[o]); frac = np.zeros_like(w)
        full = cum <= M; frac[o[full]] = 1.0
        if (~full).any():
            j = np.argmax(~full); prev = cum[j - 1] if j > 0 else 0.0
            frac[o[j]] = max(0.0, (M - prev) / w[o[j]])
        frac[s <= 0] = 0.0
        return frac, r
    def energy(xi):
        frac, r = sel(xi)
        return np.sum(w * frac * (np.sqrt(nu * vth / xi) - nu * vth / kappa)), frac, r
    lo, hi = 1e-16, (kappa ** 2 / (nu * vth)).max()
    for _ in range(120):
        mid = np.sqrt(lo * hi); e, _, _ = energy(mid)
        lo, hi = (mid, hi) if e > P else (lo, mid)
    xi = np.sqrt(lo * hi); e, frac, r = energy(xi)
    D = np.where(frac > 0, np.minimum(kappa, r), kappa)
    x = np.where(frac > 0, nu * (1 / D - 1 / kappa), 0.0)
    return xi, frac, x, D

def run_gop(L, c, z, st, fl, snr_db, n_eff, rng, use_floor=True, use_si=True, Dprev=None, margin_db=0.0):
    """one group; returns reconstructed coefficients, model distortion per chunk, flags."""
    lam, om, a, bits_meta = st
    nu_r = 0.5; g = 10 ** (snr_db / 10)
    Cc = np.log2(1 + 10 ** ((snr_db - margin_db) / 10))   # floor and metadata sized for the margined SNR
    if use_floor:
        nF = fl["bits_lap"] * 1.005 / Cc + bits_meta / Cc   # dispersion margin + metadata in the floor block
        th = fl["theta"]
        floor_term = np.where(lam > th, 1 / th - 1 / lam, 0.0)
        cF = fl["cF"]; invth = np.where(lam > th, 1 / th, 0.0); vth = np.minimum(lam, th)
    else:
        nF = bits_meta / Cc; th = np.inf; floor_term = np.zeros(L.nch); cF = np.zeros(L.k, np.float32)
        invth = np.zeros(L.nch); vth = lam.copy()
    if nF >= n_eff: return None
    if use_si:
        Dz = si_error(L, Dprev) if Dprev is not None else lam
        mu = om + a * a * Dz
    else:
        mu = lam.copy()
    mu = np.minimum(mu, lam)
    kappa = 1 / (1 / mu + floor_term)
    nA = n_eff - nF; P = nA * g; M = 2 * nA
    xi, frac, x, Dm = alloc_chunks(kappa, vth, L.n, nu_r, P, M)
    # active coefficient mask (boundary chunk partial)
    act = (frac[L.cid] >= 1.0)
    part = np.flatnonzero((frac > 0) & (frac < 1.0))
    for jb in part:
        idx = np.flatnonzero(L.cid == jb)[: int(round(frac[jb] * L.n[jb]))]; act[idx] = True
    gm = np.sqrt(x)[L.cid] * act
    r = c - cF
    s = gm * r
    sc = np.sqrt(P / max(np.sum(s.astype(np.float64) ** 2), 1e-12)); gm = gm * sc; s = s * sc
    y = s + np.sqrt(nu_r) * rng.standard_normal(L.k).astype(np.float32)
    kap_c = kappa[L.cid]; mu_c = mu[L.cid]
    invD = 1 / kap_c + (gm ** 2) / nu_r
    zz = (a[L.cid] * z) if (use_si and z is not None) else 0.0      # prior mean is a_j * z
    chat = (zz / mu_c + (invth[L.cid] + gm ** 2 / nu_r) * cF + gm / nu_r * y) / invD
    # model distortion for the recursion (per chunk, coefficient-weighted over active part)
    Dnext = np.where(frac > 0, frac * Dm + (1 - frac) * kappa, kappa)
    return dict(
        chat=chat.astype(np.float32),
        Dmodel=Dnext,
        nF=nF,
        mu=mu,
        kappa=kappa,
        frac=frac,
        Dm=Dm,
        vth=vth,
        xi=xi,
        P=P,
        M=M,
        act=act,
    )