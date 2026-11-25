import sys, os, csv, struct, numpy as np, librosa, soundfile as sf, webrtcvad
src, outdir = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 160
os.makedirs(outdir, exist_ok=True)
vad_sr=16000; y16,_=librosa.load(src,sr=vad_sr,mono=True)
win=int(0.03*vad_sr); vad=webrtcvad.Vad(2); flags=[]
for i in range(0,len(y16)-win,win):
    pcm=struct.pack('%dh'%win, *(np.clip(y16[i:i+win],-1,1)*32767).astype(np.int16))
    flags.append(vad.is_speech(pcm,vad_sr))
flags=np.array(flags,dtype=np.float32); w=8
pad=np.concatenate((np.zeros((w-1)//2),flags,np.zeros(w//2))); c=np.cumsum(pad); sm=(c[w:]-c[:-w])/w
sm=(sm>=0.5).astype(np.int8)
regions=[]; start=None; last=None
for i,v in enumerate(sm):
    t0=i*0.03; t1=t0+0.03
    if v:
        if start is None: start=t0
        last=t1
    else:
        if start is not None and t0-last>0.30:
            regions.append((start,last)); start=None
if start is not None: regions.append((start,last))
SR=32000; y,_=librosa.load(src,sr=SR,mono=True)

def split_long(a, b):
    """把超过 10s 的连续说话段按能量最低点切成 4~9s 的小段。"""
    if b - a <= 10:
        return [(a, b)]
    seg = y[int(a*SR):int(b*SR)]
    fl = int(0.05*SR); n = len(seg)//fl
    if n < 20: return [(a, b)]
    e = np.sqrt(np.mean(seg[:n*fl].reshape(n, fl)**2, axis=1))
    e = np.convolve(e, np.ones(3)/3, 'same')
    pieces = []; pos = 0.0; end = b - a
    while end - pos > 10:
        lo = int((pos+4)/0.05); hi = min(n, int((pos+9.5)/0.05))
        if hi <= lo: break
        cut = (lo + int(np.argmin(e[lo:hi]))) * 0.05
        pieces.append((a+pos, a+cut)); pos = cut
    if end - pos >= 3:
        pieces.append((a+pos, a+end))
    return pieces

cands=[]
for a0,b0 in regions:
    if a0 < 120: continue
    for a,b in split_long(a0,b0):
        d=b-a
        if d<3 or d>10: continue
        seg=y[int(a*SR):int(b*SR)]
        if len(seg)<SR*3: continue
        rms=float(np.sqrt(np.mean(seg**2))); peak=float(np.abs(seg).max())
        if rms<0.008 or rms>0.2: continue
        S=np.abs(librosa.stft(seg,n_fft=1024,hop_length=320))**2
        flat=float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))
        db=20*np.log10(rms+1e-9)
        score=(min(d,9)-3)*0.12 - flat*30 - abs(db+21)*0.06
        cands.append((score,a,b,d,rms,db,flat,peak))
cands.sort(reverse=True)
sel=cands[:limit]
rows=[]
for i,(score,a,b,d,rms,db,flat,peak) in enumerate(sel,1):
    seg=y[int(a*SR):int(b*SR)]
    if peak>0.95:
        seg = seg * (0.95/peak); peak = 0.95
    name=f'seg_{i:04d}.wav'
    sf.write(os.path.join(outdir,name),seg.astype(np.float32),SR,subtype='PCM_16')
    rows.append([name,f'{a:.2f}',f'{b:.2f}',f'{d:.2f}',f'{db:.1f}',f'{flat:.4f}',f'{peak:.3f}',f'{score:.3f}'])
with open(os.path.join(outdir,'manifest.csv'),'w',newline='') as f:
    wcsv=csv.writer(f); wcsv.writerow(['file','start_s','end_s','duration_s','rms_dbfs','flatness','peak','score']); wcsv.writerows(rows)
print('candidates',len(cands),'selected',len(sel),'total_min %.2f'%(sum(float(r[3]) for r in rows)/60))
