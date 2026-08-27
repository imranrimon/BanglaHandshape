"""Phase-1 journal-rigor analyses computed entirely from the result CSVs (no GPU).

Produces, for BdSL47:
  1. FAIRNESS / worst-signer   : per-signer mean, std, worst-signer accuracy (from LOUO).
  2. VARIANCE DECOMPOSITION     : between-signer vs between-seed variance (signer noise dominates).
  3. PAIRED SIGNIFICANCE        : pose vs appearance across the 10 LOUO signers
                                  (paired-bootstrap 95% CI, sign-flip permutation p, paired Cohen's d).
  4. IDENTITY-SHORTCUT GAP      : Top-1(SD) - Top-1(SI) with a bootstrap CI (the headline metric).

Writes results/phase1_analysis.md and results/SF5_per_signer.png.

Usage:  python tools/phase1_analysis.py --csv results/_merged.csv
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np
import pandas as pd

LOUO_RE = re.compile(
    r'^bhc_(keypoint_mlp|probe_dinov2_b)_louo_bdsl47_(digits|letters)_testuser(\d+)_seed(\d+)$')
PLAIN_RE = re.compile(r'^bhc_bdsl47_(si|sd)_bdsl47_(digits|letters)_seed(\d+)$')
MOD = {'keypoint_mlp': 'pose', 'probe_dinov2_b': 'appearance'}


def _load(csv):
    df = pd.read_csv(csv)
    df = df[df.Experiment.astype(str).str.startswith('bhc_')].copy()
    df['acc'] = df.Top1_Acc.astype(float)
    if df.acc.max() <= 1.0:
        df['acc'] *= 100.0
    return df


def _louo(df):
    rows = []
    for e, a in zip(df.Experiment, df.acc):
        m = LOUO_RE.match(str(e))
        if m:
            rows.append((MOD[m.group(1)], m.group(2), int(m.group(3)), int(m.group(4)), a))
    return pd.DataFrame(rows, columns=['modality', 'source', 'signer', 'seed', 'acc'])


def _plain(df):
    rows = []
    for e, a in zip(df.Experiment, df.acc):
        m = PLAIN_RE.match(str(e))
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3)), a))
    return pd.DataFrame(rows, columns=['split', 'source', 'seed', 'acc'])


def fairness(l):
    g = l.groupby(['modality', 'source', 'signer']).acc.mean().reset_index()
    out = []
    for (mod, src), s in g.groupby(['modality', 'source']):
        v = s.acc.values
        out.append(dict(modality=mod, source=src, n_signers=len(v),
                        mean=v.mean(), std=v.std(ddof=1), worst=v.min(),
                        worst_signer=int(s.loc[s.acc.idxmin(), 'signer']), best=v.max()))
    return pd.DataFrame(out)


def variance_decomp(l):
    out = []
    for (mod, src), s in l.groupby(['modality', 'source']):
        var_signer = s.groupby('signer').acc.mean().var(ddof=1)   # between-signer
        var_seed = s.groupby('seed').acc.mean().var(ddof=1)       # between-seed (init noise)
        out.append(dict(modality=mod, source=src, var_signer=var_signer,
                        var_seed=var_seed, ratio=var_signer / max(var_seed, 1e-9)))
    return pd.DataFrame(out)


def paired_sig(l, rng, B=10000, P=10000):
    g = l.groupby(['modality', 'source', 'signer']).acc.mean().reset_index()
    out = []
    for src in ['digits', 'letters']:
        ap = g[(g.modality == 'appearance') & (g.source == src)].set_index('signer').acc
        po = g[(g.modality == 'pose') & (g.source == src)].set_index('signer').acc
        idx = ap.index.intersection(po.index)
        d = (po.loc[idx] - ap.loc[idx]).values          # pose - appearance, per signer
        n = len(d); mean = d.mean()
        boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(B)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        signs = rng.choice([-1, 1], size=(P, n))
        perm = (signs * d).mean(axis=1)
        p = (np.sum(np.abs(perm) >= abs(mean)) + 1) / (P + 1)
        out.append(dict(source=src, n=n, mean_diff=mean, ci_lo=lo, ci_hi=hi,
                        p_value=p, cohens_d=mean / d.std(ddof=1)))
    return pd.DataFrame(out)


def shortcut_gap(p, rng, B=10000):
    out = []
    for src in ['digits', 'letters']:
        sd = p[(p.split == 'sd') & (p.source == src)].acc.values
        si = p[(p.split == 'si') & (p.source == src)].acc.values
        if len(sd) == 0 or len(si) == 0:
            continue
        gap = sd.mean() - si.mean()
        boots = np.array([rng.choice(sd, len(sd)).mean() - rng.choice(si, len(si)).mean()
                          for _ in range(B)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(dict(source=src, sd=sd.mean(), si=si.mean(), gap=gap, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/_merged.csv')
    ap.add_argument('--out', default='results/phase1_analysis.md')
    ap.add_argument('--fig', default='results/SF5_per_signer.png')
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df = _load(args.csv)
    l = _louo(df); p = _plain(df)
    if l.empty:
        sys.exit('[abort] no LOUO rows found — need bhc_louo_* results')

    fair = fairness(l); var = variance_decomp(l)
    sig = paired_sig(l, rng); gap = shortcut_gap(p, rng)

    def md(dfr, cols, fmts):
        h = '| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n'
        for _, r in dfr.iterrows():
            h += '| ' + ' | '.join(fmts[c](r) for c in cols) + ' |\n'
        return h

    L = []
    L.append('# Phase-1 rigor analyses (BdSL47, from result CSVs)\n')
    L.append('## 1. Fairness / worst-signer (LOUO, mean over 3 seeds per signer)\n')
    L.append('Per-signer accuracy statistics across the 10 leave-one-signer-out folds. '
             '`worst` is the worst-signer accuracy (a group-robustness metric); large '
             '`std` = unfair across signers.\n')
    L.append(md(fair.sort_values(['source', 'modality']),
        ['modality', 'source', 'mean', 'std', 'worst', 'worst_signer', 'best'],
        {'modality': lambda r: r.modality, 'source': lambda r: r.source,
         'mean': lambda r: f'{r["mean"]:.1f}', 'std': lambda r: f'{r["std"]:.1f}',
         'worst': lambda r: f'{r.worst:.1f}', 'worst_signer': lambda r: f'user{int(r.worst_signer):02d}',
         'best': lambda r: f'{r.best:.1f}'}))
    L.append('\n**Takeaway:** appearance is both lower and far less fair across signers '
             '(large std, low worst-signer); pose is high and stable.\n')

    L.append('\n## 2. Variance decomposition: signer noise vs seed noise\n')
    L.append('Variance of per-signer means vs variance of per-seed means. A ratio $\\gg 1$ '
             'means signer identity, not initialization, drives the variance — so LOUO '
             '(signer) spread, not seed spread, is the honest error bar.\n')
    L.append(md(var.sort_values(['source', 'modality']),
        ['modality', 'source', 'var_signer', 'var_seed', 'ratio'],
        {'modality': lambda r: r.modality, 'source': lambda r: r.source,
         'var_signer': lambda r: f'{r.var_signer:.1f}', 'var_seed': lambda r: f'{r.var_seed:.3f}',
         'ratio': lambda r: f'{r.ratio:.0f}x'}))

    L.append('\n## 3. Pose vs appearance, paired across signers\n')
    L.append('Paired difference (pose $-$ appearance) over the 10 signers: paired-bootstrap '
             '95% CI, sign-flip permutation $p$, and paired Cohen\'s $d$.\n')
    L.append(md(sig, ['source', 'n', 'mean_diff', 'ci_lo', 'ci_hi', 'p_value', 'cohens_d'],
        {'source': lambda r: r.source, 'n': lambda r: f'{int(r.n)}',
         'mean_diff': lambda r: f'{r.mean_diff:+.1f}', 'ci_lo': lambda r: f'{r.ci_lo:+.1f}',
         'ci_hi': lambda r: f'{r.ci_hi:+.1f}', 'p_value': lambda r: f'{r.p_value:.4f}',
         'cohens_d': lambda r: f'{r.cohens_d:.2f}'}))

    if not gap.empty:
        L.append('\n## 4. Identity-shortcut gap (SD $-$ SI, DINOv2-S+LoRA)\n')
        L.append(md(gap, ['source', 'sd', 'si', 'gap', 'ci_lo', 'ci_hi'],
            {'source': lambda r: r.source, 'sd': lambda r: f'{r.sd:.1f}',
             'si': lambda r: f'{r.si:.1f}', 'gap': lambda r: f'{r.gap:.1f}',
             'ci_lo': lambda r: f'{r.ci_lo:.1f}', 'ci_hi': lambda r: f'{r.ci_hi:.1f}'}))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L))
    print('wrote', args.out)

    # figure: per-signer accuracy, pose vs appearance
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        g = l.groupby(['modality', 'source', 'signer']).acc.mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, src in zip(axes, ['digits', 'letters']):
            po = g[(g.modality == 'pose') & (g.source == src)].sort_values('signer')
            ap = g[(g.modality == 'appearance') & (g.source == src)].sort_values('signer')
            x = np.arange(len(po)); w = 0.4
            ax.bar(x - w/2, ap.acc, w, label='appearance', color='#c44')
            ax.bar(x + w/2, po.acc, w, label='pose', color='#4a8')
            ax.set_xticks(x); ax.set_xticklabels([f'{int(s):02d}' for s in po.signer], fontsize=7)
            ax.set_title(f'BdSL47-{src}: per held-out signer'); ax.set_xlabel('test signer')
            ax.set_ylabel('Top-1 (%)'); ax.set_ylim(0, 105); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(args.fig, dpi=150); print('wrote', args.fig)
    except Exception as e:
        print('[warn] figure skipped:', e)


if __name__ == '__main__':
    main()
