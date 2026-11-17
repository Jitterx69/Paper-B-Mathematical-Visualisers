import numpy as np, json, time, sys
from evallab import *
H,W=352,640; NG=16
P=load_yuv('bbb352.yuv',H,W,NG*G)
L=Layout([(H,W),(H//2,W//2),(H//2,W//2)])
Tg=G/25; scale=(352*640)/(360*640)
Bc=500e3*Tg*scale; n_eff=2.5e6*Tg/np.log2(1+10)*scale
def coefs(g): return np.concatenate([fwd(p[g*G:(g+1)*G]).ravel() for p in P]).astype(np.float32)
C=[coefs(g) for g in range(NG)]
def z_orig(g): return np.concatenate([si_plane(p[g*G-1]).ravel() for p in P]).astype(np.float32)
ST=[];FL=[]
for g in range(NG):
    z0=z_orig(g) if g>0 else None
    st=gop_stats(L,C[g],z0); ST.append(st); FL.append(floor_design(L,st[0],st[1],st[2],Bc,C[g]))
info=dict(Bc=Bc,n_eff=n_eff,k=L.k,theta=[f['theta'] for f in FL],DF=[f['DF'] for f in FL],
          mse_floor=[f['mse_floor'] for f in FL],bits_lap=[f['bits_lap'] for f in FL],Rintra=[f['Rintra'] for f in FL],
          meta=[float(s[3]) for s in ST],nF_frac=[f['nF_coef']/L.k for f in FL],delta=[f['delta'] for f in FL])
json.dump(info,open('eval_info.json','w'),default=float)
print("setup done",flush=True)
def recon_last(chat):
    out=[];p=0
    for pl,s in zip(P,L.sizes):
        out.append(last_frame(chat[p:p+s].reshape((G,pl.shape[1]//8,pl.shape[2]//8,8,8)))); p+=s
    return out
def simulate(snr,use_floor,use_si,seed=0,margin=0.0):
    rng=np.random.default_rng(seed); Dprev=None; zrec=None
    psnr=[];model=[];nfr=[]
    for g in range(NG):
        z=None
        if use_si and zrec is not None: z=np.concatenate([si_plane(f).ravel() for f in zrec]).astype(np.float32)
        st=ST[g]
        if g==0 or not use_si: st=(st[0],st[1],np.zeros_like(st[2]),st[3])
        out=run_gop(L,C[g],z,st,FL[g],snr,n_eff,rng,use_floor,use_si,Dprev,margin)
        if out is None: return None
        mse=float(np.mean((out['chat']-C[g])**2)); psnr.append(10*np.log10(255**2/mse))
        model.append(10*np.log10(255**2/(np.sum(L.n*out['Dmodel'])/L.k))); nfr.append(out['nF']/n_eff)
        Dprev=out['Dmodel']
        if use_si: zrec=[np.clip(f,0,255) for f in recon_last(out['chat'])]
    return dict(psnr=float(np.mean(psnr)),psnr_min=float(np.min(psnr)),model=float(np.mean(model)),nF=float(np.mean(nfr)))
mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
if mode == "grid":
    res = {}; t = time.time()
    for name, (uf, us) in {"ours": (True, True), "ours_noSI": (True, False),
                           "nofloor_SI": (False, True), "softcast": (False, False)}.items():
        for snr in [2, 4, 6, 8, 10, 12]:
            res[f"{name}|{snr}"] = simulate(snr, uf, us)
            print(name, snr, res[f"{name}|{snr}"], round(time.time() - t), flush=True)
    json.dump(res, open('eval_res.json', 'w'))
elif mode == "margin":
    res = {f"ours_M3|{snr}": simulate(snr, True, True, margin=3.0) for snr in [8, 10, 12]}
    print(res); json.dump(res, open('eval_margin.json', 'w'))
elif mode == "floor":
    ps = [10 * np.log10(255 ** 2 / m) for m in info['mse_floor']]
    bl = np.array(info['bits_lap']); ri = np.array(info['Rintra']); me = np.array(info['meta'])
    print("floor-only PSNR mean/min", np.mean(ps), np.min(ps), "theta", np.mean(info['theta']),
          "|F|/k", np.mean(info['nF_frac']))
    print("bits_lap", bl.mean(), "Rintra", ri.mean(), "meta", me.mean(), "Bc", Bc,
          "B_F/Bc", ((bl + me) / Bc).mean(), "max", (bl + me).max())