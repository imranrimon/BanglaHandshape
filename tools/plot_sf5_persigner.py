"""SF5 — per-held-out-signer LOUO Top-1: appearance (DINOv2-B) vs pose (keypoint MLP),
BdSL47 Digits + Letters. Reads the fixed-data LOUO CSVs, de-dups to the latest run per
(source, signer, seed), averages over seeds. Clean fonts, 600 dpi, PDF+PNG."""
import csv, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 12})
APP="#c1443c"; POSE="#2a9d8f"
SRC={"appearance":"results/bhc_louo_appearance.csv","pose":"results/bhc_louo_keypoint.csv"}

def load(path):
    latest={}   # (source,user,seed) -> (timestamp, acc)
    for r in csv.DictReader(open(path)):
        m=re.search(r"louo_bdsl47_(digits|letters)_testuser(\d+)_seed(\d+)", r["Experiment"])
        if not m: continue
        k=(m.group(1),int(m.group(2)),int(m.group(3)))
        ts=r["Timestamp"]
        if k not in latest or ts>latest[k][0]:
            latest[k]=(ts,float(r["Top1_Acc"])*100)
    agg={}      # (source,user) -> mean over seeds
    for (src,u,s),(ts,a) in latest.items():
        agg.setdefault((src,u),[]).append(a)
    return {k:np.mean(v) for k,v in agg.items()}

data={name:load(p) for name,p in SRC.items()}
users=list(range(1,11))
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,src in zip(axes,("digits","letters")):
    app=[data["appearance"].get((src,u),np.nan) for u in users]
    pos=[data["pose"].get((src,u),np.nan) for u in users]
    x=np.arange(len(users)); w=0.4
    ax.bar(x-w/2,app,w,label="Appearance (DINOv2-B)",color=APP,edgecolor="black",linewidth=0.5)
    ax.bar(x+w/2,pos,w,label="Pose (keypoint MLP)",color=POSE,edgecolor="black",linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{u:02d}" for u in users])
    ax.set_xlabel("Held-out test signer"); ax.set_ylim(0,105)
    ax.set_title(f"BdSL47-{src.capitalize()}",fontsize=14,pad=6)
    ax.grid(axis="y",alpha=0.3)
    print(f"  {src}: appearance mean={np.nanmean(app):.1f} pose mean={np.nanmean(pos):.1f}")
axes[0].set_ylabel("Top-1 (%)")
axes[0].legend(loc="lower left",framealpha=0.9)
fig.tight_layout()
os.makedirs("paper/figs",exist_ok=True)
for ext in ("pdf","png"):
    fig.savefig(f"paper/figs/SF5_per_signer.{ext}",dpi=600,bbox_inches="tight")
print("wrote paper/figs/SF5_per_signer.{pdf,png} @600dpi")
