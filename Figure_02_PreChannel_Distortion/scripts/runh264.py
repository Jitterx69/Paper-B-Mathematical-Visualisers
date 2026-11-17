import numpy as np, subprocess, json, os, imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe(); H,W=352,640; NF=128; Tg=8/25
n_eff=2.5e6*Tg/np.log2(11)*(352*640)/(360*640)
src=np.fromfile('bbb352.yuv',dtype=np.uint8).reshape(NF,-1).astype(np.float32)
def run(cmd): subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def enc(rate,mode):
    extra=[] if mode=="medium" else ["-tune","zerolatency","-g","8","-x264-params","keyint=8:min-keyint=8:scenecut=0"]
    base=[FF,"-y","-f","rawvideo","-pix_fmt","yuv420p","-s",f"{W}x{H}","-r","25","-i","bbb352.yuv","-c:v","libx264","-preset","medium","-b:v",str(int(rate)),"-an"]+extra
    run(base+["-pass","1","-passlogfile","pl","-f","null","/dev/null"])
    run(base+["-pass","2","-passlogfile","pl","out.mp4"])
    run([FF,"-y","-i","out.mp4","-f","rawvideo","-pix_fmt","yuv420p","dec.yuv"])
    dec=np.fromfile('dec.yuv',dtype=np.uint8).reshape(NF,-1).astype(np.float32)
    ach=os.path.getsize('out.mp4')*8/(NF/25)
    mse=((dec-src)**2).mean(axis=1).reshape(16,8).mean(axis=1)   # per group, combined YUV samples
    return float(np.mean(10*np.log10(255**2/np.maximum(mse,1e-9)))), float(np.min(10*np.log10(255**2/np.maximum(mse,1e-9)))), ach
if __name__ == "__main__":
    res={}
    for snr in [2,4,6,8,10,12]:
        R=n_eff*np.log2(1+10**(snr/10))/Tg
        for mode in ["medium","zerolatency"]:
            p,pm,ach=enc(R,mode); res[f"{mode}|{snr}"]=dict(target=R,achieved=ach,psnr=p,psnr_min=pm)
            print(mode,snr,round(R/1e3),round(ach/1e3),round(p,2),round(pm,2),flush=True)
    json.dump(res,open('h264_res.json','w'))
    