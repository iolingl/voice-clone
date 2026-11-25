import sys, struct, csv, numpy as np, librosa, soundfile as sf, webrtcvad, os
src = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
vad_sr = 16000
y16, _ = librosa.load(src, sr=vad_sr, mono=True)
win = int(0.03*vad_sr)
vad = webrtcvad.Vad(2)
flags=[]
for i in range(0, len(y16)-win, win):
    pcm=struct.pack('%dh'%win, *(np.clip(y16[i:i+win],-1,1)*32767).astype(np.int16))
    flags.append(vad.is_speech(pcm, vad_sr))
flags=np.array(flags,dtype=np.float32)
w=8; pad=np.concatenate((np.zeros((w-1)//2),flags,np.zeros(w//2))); c=np.cumsum(pad); sm=(c[w:]-c[:-w])/w
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
# load 24k for export
y24,_ = librosa.load(src, sr=24000, mono=True)
cands=[]
for a,b in regions:
    dur=b-a
    if dur<8 or dur>20 or a<120: continue
    seg=y24[int(a*24000):int(b*24000)]
    if len(seg)==0: continue
    rms=float(np.sqrt(np.mean(seg**2))); peak=float(np.abs(seg).max())
    if peak>0.98 or rms<0.005 or rms>0.25: continue
    S=np.abs(librosa.stft(seg,n_fft=1024,hop_length=240))**2
    flat=librosa.feature.spectral_flatness(S=S)[0]
    # score: prefer 10-18s, low flatness, rms near -20 dBFS
    db=20*np.log10(rms+1e-9)
    score=(min(dur,18)-8)*0.15 - float(np.mean(flat))*30 - abs(db+20)*0.08
    cands.append((score,a,b,dur,rms,db,float(np.mean(flat)),peak))
cands.sort(reverse=True)
print('candidates:',len(cands))
rows=[]
for i,(score,a,b,dur,rms,db,flat,peak) in enumerate(cands[:12],1):
    seg=y24[int(a*24000):int(b*24000)]
    name=f'ref_{i:02d}_{a:.1f}s.wav'
    sf.write(os.path.join(outdir,name), seg.astype(np.float32), 24000, subtype='PCM_16')
    rows.append([name,f'{a:.2f}',f'{b:.2f}',f'{dur:.2f}',f'{db:.1f}',f'{flat:.4f}',f'{peak:.3f}',f'{score:.3f}'])
    print(rows[-1])
with open(os.path.join(outdir,'references.csv'),'w',newline='') as f:
    wcsv=csv.writer(f); wcsv.writerow(['file','start_s','end_s','duration_s','rms_dbfs','flatness','peak','score']); wcsv.writerows(rows)
