import sys, struct, numpy as np, librosa, webrtcvad
p = sys.argv[1]
sr = 16000
y, _ = librosa.load(p, sr=sr, mono=True)
win = int(0.03*sr)
vad = webrtcvad.Vad(2)
flags = []
for i in range(0, len(y)-win, win):
    pcm = struct.pack('%dh'%win, *(np.clip(y[i:i+win],-1,1)*32767).astype(np.int16))
    flags.append(vad.is_speech(pcm, sr))
flags = np.array(flags, dtype=np.float32)
# smooth
w=8
pad=np.concatenate((np.zeros((w-1)//2),flags,np.zeros(w//2)))
c=np.cumsum(pad); sm=(c[w:]-c[:-w])/w
sm=(sm>=0.5).astype(np.int8)
# contiguous regions of speech, merge gaps < 0.30s
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
durs=np.array([b-a for a,b in regions])
print('audio dur %.1f s, speech ratio %.1f%%, regions %d'%(len(y)/sr, 100*flags.mean(), len(regions)))
for lo,hi in [(1,3),(3,5),(5,8),(8,10),(10,15),(15,20),(20,30),(30,999)]:
    m=(durs>=lo)&(durs<hi)
    print(f'{lo:>3}-{hi:<3}s: {m.sum():4d} regions, {durs[m].sum()/60:7.2f} min')
print('total speech in regions: %.2f min'%(durs.sum()/60))
# print top 10 longest
idx=np.argsort(durs)[::-1][:10]
for i in idx: print('  %7.1f-%7.1f  %.2fs'%(regions[i][0],regions[i][1],durs[i]))
