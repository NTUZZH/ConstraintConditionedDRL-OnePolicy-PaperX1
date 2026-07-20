"""this paper figures (F2 parity bars, F3 zero-shot heatmap, F4 causality).

Each figure is generated AT ITS PLACED PHYSICAL SIZE (IEEE single column =
3.5 in) with fonts >= 7 pt, then rendered to PNG for manual inspection and
PDF for inclusion. Colors: Okabe-Ito CVD-safe palette (Wong, Nature Methods
2011), FIXED entity->hue mapping (never cycled); the heatmap uses a
single-hue sequential ramp with direct cell labels (identity/value never
color-alone).

Usage: python scripts/make_family_figures.py [--only f2|f3|f4]
Missing inputs are skipped with a message (rerun as results land).
"""
import glob
import json
import os
import re
import sys

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator

RES = 'test_results/FAMILY/10x25'
OUT = 'paper/figures'

# fixed entity -> color (Okabe-Ito)
C_JOINT = '#0072B2'      # blue
C_SPEC = '#E69F00'       # orange
C_BLIND = '#999999'      # gray
C_PDR = '#009E73'        # green

plt.rcParams.update({
    # Times New Roman throughout. Liberation Serif is the metric-compatible
    # TrueType clone of Times New Roman and is what fc-match typically
    # resolves "Times New Roman" to, so it goes FIRST: Nimbus Roman
    # is an OpenType/CFF face that matplotlib embeds as /CIDFontType2 with a
    # /FontFile2 payload, which makes every figure PDF raise "Mismatch between
    # font type and embedded font file" in pdffonts and risks an IEEE PDF check.
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'Nimbus Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'pdf.fonttype': 42,
})

# imshow() writes a RASTER image into the PDF, always. On a heat map this small it
# lands as a 264 ppi indexed bitmap: below IEEE's 300 dpi floor for images, and a
# bitmap in a figure that is otherwise pure vector. The cells are rectangles, so we
# draw rectangles. Same colours, same mapping, no raster object in the output.
def _cells(ax, shade, cmap='Blues', vmin=0.0, vmax=1.15):
    """Draw a heat map as vector rectangles, one per cell, in imshow's coordinates
    (cell (r,c) is centred on (c, r) and spans one unit in each direction)."""
    cm = plt.get_cmap(cmap)
    nr, nc = shade.shape
    for r in range(nr):
        for c in range(nc):
            v = (shade[r, c] - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=cm(float(np.clip(v, 0, 1))),
                                   edgecolor='none', linewidth=0, zorder=0))
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)          # imshow's origin is top-left
    ax.set_aspect('auto')




def _stale(name, why):
    """Overwrite a figure whose inputs are missing with a loud placeholder.

    The placeholder keeps the document compiling while making the figure
    impossible to mistake for a result: a stale .pdf left on disk would keep
    rendering silently, and a deleted one would break the build.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.10), dpi=300)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, ec='#D55E00', lw=1.5))
    ax.text(0.5, 0.62, 'FIGURE NOT REGENERATED', ha='center', va='center',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color='#D55E00')
    ax.text(0.5, 0.32, why, ha='center', va='center', wrap=True,
            transform=ax.transAxes, fontsize=7, color='#D55E00')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  PLACEHOLDER written for {name}: {why}')

def _load(tag, regime):
    """Per-instance makespans for one method on one regime.

    Result files come in two shapes: [n, 2] (makespan, seconds) from the policy
    evaluations, and [n] from methods that record no timing, so the load guards
    on ndim before taking column 0.
    """
    p = os.path.join(RES, f'Result_{tag}_{regime}.npy')
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def _pooled(base, regime):
    """Per-instance makespan of a trained model, meaned across its three seeds.

    EVERY learned policy in a figure goes through this. Nothing may reach a panel
    through the bare _load(), because _load() returns seed 301 alone, and a panel
    that pools three seeds on one side of a comparison and one on the other is
    measuring its estimator rather than its policies. That is not hypothetical:
    panel (b) of f23 did exactly this. The joint row was pooled and the
    specialist and token-blind rows were not, so the baseline was represented by
    a single draw -- which on BOTH held-out compositions happened to be that
    baseline's worst seed. The figure showed the joint beating the SW specialist
    while the paper's own E2 macro (\\EqSwDelta) reported it 0.2% behind.

    Falls back to whatever seeds exist, and says so, rather than silently
    comparing unequal pools.
    """
    seeds = [base, f'{base}-s302', f'{base}-s303']
    arrs = [_load(f'greedy+{m}', regime) for m in seeds]
    have = [s for s, a in zip(seeds, arrs) if a is not None]
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    if len(arrs) < 3:
        print(f'  seed note: {base} on {regime} has {len(arrs)}/3 seeds '
              f'({", ".join(h.split("+")[-1] for h in have)})')
    return np.mean(arrs, axis=0)


def _load_joint(model, regime):
    return _pooled(model, regime)


def _load_spec(regime):
    return _pooled(f'10x25+family+spec-{regime}', regime)


def _paired_boot(d, n=20000, seed=7):
    """95% CI of the mean paired gap. Same estimator as p11_equivalence.py."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f2_parity(joint='10x25+family+joint-v1', margin=1.0):
    """Forest plot of the PAIRED relative gap, joint minus specialist (E1).

    Shows exactly the statistic the parity claim rests on: E1 is a PAIRED
    equivalence test against a +-1% margin, so the figure plots one point per
    regime, its paired-bootstrap 95% CI, and the equivalence band. It reads
    the same per-seed arrays as p11_equivalence.py, so figure and macros
    cannot drift.
    """
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW']
    los, his, mids = [], [], []
    for r in regimes:
        j = _load_joint(joint, r)
        s = _load_spec(r)
        if j is None or s is None:
            print(f'F2: missing inputs for {r}')
            _stale('f2_parity', f'joint or specialist results missing for regime {r}')
            return
        d = 100.0 * (j - s) / s          # per-instance relative gap, %
        lo, hi = _paired_boot(d)
        los.append(lo); his.append(hi); mids.append(float(d.mean()))

    # Three verdicts, three colours. An interval wholly inside the band is
    # EQUIVALENT; one wholly beyond it is INFERIOR, a decided loss; one that
    # straddles the band edge is INCONCLUSIVE. Drawing the last two alike would
    # let a reader take a decided loss for an undecided one.
    def _verdict(lo, hi):
        if -margin < lo and hi < margin:
            return 'equivalent'
        if lo > margin:
            return 'inferior'
        return 'inconclusive'

    STYLE = {'equivalent':   (C_JOINT,   'o'),
             'inconclusive': ('#999999', 's'),
             'inferior':     ('#D55E00', 'D')}

    y = np.arange(len(regimes))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.10), dpi=300)
    ax.axvspan(-margin, margin, color=C_SPEC, alpha=0.16, lw=0,
               label=f'$\\pm${margin:g}% margin')
    ax.axvline(0, color='0.35', lw=0.6, zorder=1)
    seen = set()
    for yi, lo, hi, m in zip(y, los, his, mids):
        v = _verdict(lo, hi)
        c, mk = STYLE[v]
        ax.plot([lo, hi], [yi, yi], lw=1.4, color=c, solid_capstyle='round',
                zorder=3)
        ax.plot([m], [yi], mk, ms=3.4, color=c, zorder=4,
                label=v if v not in seen else None)
        seen.add(v)
    ax.set_yticks(y); ax.set_yticklabels(regimes)
    ax.set_ylim(-0.7, len(regimes) - 0.3)
    ax.set_xlabel('paired makespan gap, joint $-$ specialist (%)')
    ax.set_ylabel('constraint regime')
    ax.grid(axis='x', lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1.2,
              columnspacing=1.0, bbox_to_anchor=(0, 1.30))
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f2_parity.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F2 written')


def f3_zeroshot(joint='10x25+family+joint-v1',
                blind='10x25+family+jointblind-v1'):
    """Heatmap: methods x held-out regimes, mean makespan (E2, money fig)."""
    rows = [('zero-shot joint', f'greedy+{joint}', None),
            ('token-blind joint', f'greedy+{blind}', None),
            ('composition specialist', 'greedy+10x25+family+spec-{R}', None),
            ('best PDR', None, ('FIFO', 'MOR', 'SPT', 'MWKR'))]
    regimes = ['SW', 'LSW']
    M = np.full((len(rows), len(regimes)), np.nan)
    for c, reg in enumerate(regimes):
        for r, (label, tag, pdrs) in enumerate(rows):
            if pdrs:
                vals = [_load(p, reg) for p in pdrs]
                vals = [v.mean() for v in vals if v is not None]
                M[r, c] = min(vals) if vals else np.nan
            elif label == 'zero-shot joint':
                v = _load_joint(joint, reg)
                M[r, c] = v.mean() if v is not None else np.nan
            else:
                v = _load(tag.replace('{R}', reg), reg)
                M[r, c] = v.mean() if v is not None else np.nan
    if np.isnan(M).any():
        print('F3: missing inputs')
        _stale('f3_zeroshot', 'one or more methods have no results on SW/LSW')
        return
    # per-COLUMN normalized shading: absolute makespans differ across
    # regimes, and the quantity being compared is the within-regime ranking of
    # methods, so each column carries its own scale (lighter = better). Shading
    # is therefore not comparable ACROSS columns; the cell labels are, and the
    # best cell per column is bolded.
    shade = np.zeros_like(M)
    for c in range(M.shape[1]):
        col = M[:, c]
        lo, hi = np.nanmin(col), np.nanmax(col)
        shade[:, c] = (col - lo) / (hi - lo + 1e-9)
    fig, ax = plt.subplots(figsize=(2.6, 1.9), dpi=300)
    _cells(ax, shade, 'Blues', 0, 1.15)
    for c in range(M.shape[1]):
        best = np.nanargmin(M[:, c])
        for r in range(M.shape[0]):
            if not np.isnan(M[r, c]):
                dark = shade[r, c] > 0.6
                ax.text(c, r, f'{M[r, c]:.1f}', ha='center', va='center',
                        fontsize=7.5,
                        fontweight='bold' if r == best else 'normal',
                        color='white' if dark else '#1a1a1a')
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([f'{r}\n(held out)' for r in regimes])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f3_zeroshot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F3 written')


def f4_causality(model='10x25+family+joint-v1'):
    """Token interventions: mean degradation vs true token per regime (E3)."""
    regimes = ['L', 'S', 'W', 'LS', 'LW']
    CF = {'L': 'S', 'S': 'L', 'W': 'LS', 'LS': 'W', 'LW': 'S'}
    CH = {'L': 'lag', 'S': 'setup', 'W': 'window', 'LS': 'lag-setup',
          'LW': 'lag-window'}
    # The third label wraps onto two lines so the frameless upper-left legend
    # ends left of the W group's error whisker instead of printing across it.
    modes = [('withhold', C_BLIND, 'zero token'),
             ('counterfactual-{CF}', C_SPEC, 'conflicting token'),
             ('withhold+chblind-{CH}', C_JOINT, 'token + channels\nremoved')]
    fig, ax = plt.subplots(figsize=(3.5, 1.52), dpi=300)
    x = np.arange(len(regimes))
    w = 0.26
    plotted = False
    for k, (mode, color, label) in enumerate(modes):
        deltas, errs = [], []
        for r in regimes:
            base = _load(f'greedy+{model}', r)
            tag = mode.replace('{CF}', CF[r]).replace('{CH}', CH[r])
            pert = _load(f'greedy+{model}+{tag}', r)
            if base is None or pert is None:
                deltas.append(np.nan); errs.append(0)
                continue
            d = pert - base
            deltas.append(d.mean())
            errs.append(1.96 * d.std() / np.sqrt(len(d)))
            plotted = True
        ax.bar(x + (k - 1) * w, deltas, w, yerr=errs, color=color,
               error_kw=dict(lw=0.7), label=label)
    if not plotted:
        print('F4: no inputs yet, skipping')
        plt.close(fig)
        return
    ax.axhline(0, color='0.35', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_ylabel('$\\Delta$ makespan vs\ntrue token (h)', fontsize=8.5)
    ax.set_xlabel('constraint regime', fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    # The only tall bar is W (centre); L and LW ends are short, so the upper-left
    # corner is clear. Keep the legend small and pinned there.
    leg = ax.legend(loc='upper left', fontsize=6.3, handlelength=0.9,
                    handletextpad=0.4, labelspacing=0.25, borderpad=0.3,
                    framealpha=0.92, edgecolor='0.75')
    leg.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f4_causality.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F4 written')


# --- E5 anytime frontier -----------------------------------------------------
# These three helpers replicate scripts/p30_anytime.py exactly so the figure and
# macros_anytime.tex cannot drift: same result files, same credited min(warm,
# search) rule, same per-instance relative-gap averaging. Do not "improve" the
# aggregation here; p30 is the generator and this only redraws its numbers.
OR = 'or_solution/FAMILY'
ANY_JOINT = 'greedy+10x25+family+joint-v1'
ANY_SEEDS = ['', '-s302', '-s303']         # p30's three seeds
ANY_REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
ANY_SETUP = ['S', 'LS', 'SW', 'LSW']       # the setup-bearing regimes
ANY_RUNGS = [1, 2, 5, 10, 30]              # 300 s is a partial prefix; excluded
ANY_WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty'}
ANY_RERUN = (5, 30)                        # tiers re-solved from the repaired hint
# The repaired best-of-four warm-start cache (scripts/p41_repaired_warm_cache.py):
# the 5s and 30s rungs were re-solved from it, the 1s/2s/10s rungs have their
# credited value floored at it. Mirrors p30_anytime exactly.
_ANY_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def _any_rung_dir(budget):
    return (f'{OR}/10x25_fairwarmR{budget}s' if budget in ANY_RERUN
            else f'{OR}/10x25_fairwarm{budget}s')


def _any_credited(budget, regime):
    """Per-instance CREDITED solver makespans against the repaired hint, for one
    (budget, regime). Exactly p30_anytime.cells() without the improved-flag."""
    out = []
    rerun = budget in ANY_RERUN
    for p in sorted(glob.glob(f'{_any_rung_dir(budget)}/{regime}/*.json')):
        d = json.load(open(p))
        x = d[0] if isinstance(d, list) else d
        obj, warm = x['objective'], x.get('warm_makespan')
        if warm is None:
            return None                    # not a fair-warm run; treat as missing
        if rerun:
            out.append(min(obj, warm))     # warm_makespan is the repaired hint
        else:
            hint = _ANY_REPAIRED[regime][x['instance']]['repaired_ms']
            out.append(min(hint, min(obj, warm)))
    return np.array(out, dtype=float) if out else None


def _any_policy(regime):
    """Per-instance greedy-policy makespan, meaned across p30's three seeds."""
    arrs = []
    for s in ANY_SEEDS:
        p = f'{RES}/Result_{ANY_JOINT}{s}_{regime}.npy'
        if not os.path.exists(p):
            return None
        a = np.load(p)
        arrs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return np.mean(arrs, axis=0)


def f5_anytime():
    """Line plot of the policy's lead over the credited fair-warm CP-SAT solver as
    the solver's time budget grows (E5). One line per regime; y is the mean
    per-instance relative gap in percent (positive = policy ahead), the same
    quantity macros_anytime.tex reports; x is the budget at the five measured
    rungs (1, 2, 5, 10, 30 s), log-scaled.

    The setup-bearing regimes (S, LS, SW, LSW) are drawn as saturated solid lines
    and the rest (N, L, W, LW) as muted dashed lines, because the story splits on
    exactly that line: where setups are active the search returns nothing better
    than its warm start until after 5 s, so those lines stay flat and positive,
    then dip toward zero by 30 s.
    """
    # A rung is only admissible if all 8 regimes x 100 instances are present, the
    # same 800-cell completeness gate p30 enforces before it will report a rung.
    for b in ANY_RUNGS:
        if len(glob.glob(f'{_any_rung_dir(b)}/*/*.json')) != 800:
            _stale('f5_anytime', f'fair-warm rung {b}s is incomplete '
                                 f'(need 800 cells: 8 regimes x 100 instances)')
            return
    if _any_policy('N') is None:
        _stale('f5_anytime', 'joint policy results (joint-v1, 3 seeds) missing')
        return

    # Compute the plotted grid: lead[(rung, regime)] = mean per-instance relative
    # gap (%), positive = policy ahead.
    lead = {}
    for r in ANY_REGIMES:
        pol = _any_policy(r)
        for b in ANY_RUNGS:
            sol = _any_credited(b, r)
            lead[(b, r)] = float((100.0 * (sol - pol) / pol).mean())

    # CORRECTNESS GATE. macros_anytime.tex is p30's output and is ground truth;
    # every plotted value must reproduce its \Any<Word><Regime>Lead macro to two
    # decimals. A drifted result file or a changed aggregation dies here rather
    # than shipping a dishonest figure.
    txt = open('paper/macros_anytime.tex').read()
    for b in ANY_RUNGS:
        for r in ANY_REGIMES:
            m = re.search(r'\\newcommand\{\\Any' + ANY_WORD[b] + r
                          + r'Lead\}\{([+-][0-9.]+)\}', txt)
            if m is None:
                raise SystemExit(f'f5: macro \\Any{ANY_WORD[b]}{r}Lead not found '
                                 f'in macros_anytime.tex')
            macro = float(m.group(1))
            if abs(round(lead[(b, r)], 2) - macro) > 0.005:
                raise SystemExit(
                    f'f5 REFUSING: {r} at {b}s computes {lead[(b, r)]:+.2f}% but '
                    f'macros_anytime.tex says {macro:+.2f}%. The figure and the '
                    f'paper disagree; fix the input, do not ship the figure.')

    # Fixed entity -> (colour, marker) mapping, Okabe-Ito, never cycled. Setup
    # regimes get saturated hues and solid lines; the rest get muted hues and a
    # dashed stroke, so the two families separate by dash even in greyscale, and
    # every regime is told apart by its marker without relying on colour.
    STYLE = {
        'S':   ('#D55E00', 'o', '-'),      # vermillion
        'LS':  ('#0072B2', 's', '-'),      # blue
        'SW':  ('#009E73', '^', '-'),      # bluish green
        'LSW': ('#CC79A7', 'D', '-'),      # reddish purple
        'N':   ('#000000', 'v', (0, (4, 2))),
        'L':   ('#E69F00', 'X', (0, (4, 2))),   # orange
        'W':   ('#56B4E9', 'P', (0, (4, 2))),   # sky blue
        'LW':  ('#999999', '*', (0, (4, 2))),   # grey
    }

    # Sized to print at one full \columnwidth (3.5 in): the canvas IS the print
    # size, so every font below lands on paper at its stated point size.
    fig, ax = plt.subplots(figsize=(3.5, 1.50), dpi=300)
    ax.axhline(0, color='0.35', lw=0.6, zorder=1)   # solver and policy tied
    # Draw the muted dashed family first so the saturated setup lines sit on top.
    order = ['N', 'L', 'W', 'LW', 'S', 'LS', 'SW', 'LSW']
    for r in order:
        c, mk, ls = STYLE[r]
        setup = r in ANY_SETUP
        # A white marker edge (halo) keeps each marker crisply separable from its
        # neighbours in greyscale, including where two solid setup curves sit on
        # top of each other in the flat region: the shape still reads even when the
        # hues are close. Colours, the per-regime marker, and the solid(setup)/
        # dashed(rest) convention are all unchanged.
        ax.plot(ANY_RUNGS, [lead[(b, r)] for b in ANY_RUNGS],
                ls=ls, color=c, marker=mk, ms=4.4, mew=0.6,
                markeredgecolor='white',
                lw=1.5 if setup else 1.2, alpha=1.0 if setup else 0.85,
                zorder=4 if setup else 3)

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ANY_RUNGS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(b) for b in ANY_RUNGS]))
    ax.xaxis.set_minor_locator(FixedLocator([]))    # no 3,4,6,... minor ticks
    ax.set_xlim(0.85, 35)
    ax.set_ylim(-6.4, 3.0)   # headroom for the in-axes legend band, top right
    ax.set_xlabel('solver time budget (s), log scale')
    ax.set_ylabel('lead over credited\nsolver (%)')
    ax.grid(axis='y', lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    # Four-column legend band in the empty top-right corner. Legend entries fill
    # column-wise, so this order puts the setup family on the TOP row and its
    # setup-free counterpart directly below it: each column differs by exactly S.
    # The old lower-left box sat on the L and LW curves' opening segments.
    from matplotlib.lines import Line2D
    leg_order = ['S', 'N', 'LS', 'L', 'SW', 'W', 'LSW', 'LW']
    handles = [Line2D([0], [0], color=STYLE[r][0], marker=STYLE[r][1],
                      ls=STYLE[r][2], lw=1.5 if r in ANY_SETUP else 1.2,
                      ms=4.4, mew=0.6, markeredgecolor='white', label=r)
               for r in leg_order]
    leg = ax.legend(handles=handles, ncol=4, loc='upper right', fontsize=6.3,
                    handlelength=1.0, handletextpad=0.4, columnspacing=0.6,
                    labelspacing=0.25, borderpad=0.3, borderaxespad=0.25,
                    framealpha=0.92, edgecolor='0.75')
    leg.get_frame().set_linewidth(0.4)

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f5_anytime.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F5 written (macro assertion passed for all '
          f'{len(ANY_RUNGS) * len(ANY_REGIMES)} regime x rung cells)')


def f23_parity_zeroshot(joint='10x25+family+joint-v1',
                        blind='10x25+family+jointblind-v1', margin=1.0):
    """E1 and E2 in one double-column float: parity forest, then zero-shot ranks.

    The two panels answer one question in two steps: does one policy match the
    specialists on the regimes it trained on (a), and does it still match them
    on compositions it never saw (b). Sharing a float saves the page budget a
    caption and a float separation, and the panels are read together anyway.
    """
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW']
    los, his, mids = [], [], []
    for r in regimes:
        j = _load_joint(joint, r)
        sp = _load_spec(r)
        if j is None or sp is None:
            _stale('f23_parity_zeroshot', f'joint or specialist missing for {r}')
            return
        d = 100.0 * (j - sp) / sp
        lo, hi = _paired_boot(d)
        los.append(lo); his.append(hi); mids.append(float(d.mean()))

    def _verdict(lo, hi):
        if -margin < lo and hi < margin:
            return 'equivalent'
        if lo > margin:
            return 'inferior'
        return 'inconclusive'

    STYLE = {'equivalent':   (C_JOINT,   'o'),
             'inconclusive': ('#999999', 's'),
             'inferior':     ('#D55E00', 'D')}

    # Model bases, NOT 'greedy+...' tags: every learned row is seed-pooled through
    # _pooled(), so panel (b) compares three seeds against three seeds, as its
    # caption says and as panel (a) already did. The rules carry no seed.
    rows = [('zero-shot joint', joint, None),
            ('token-blind joint', blind, None),
            ('composition specialist', '10x25+family+spec-{R}', None),
            ('best PDR', None, ('FIFO', 'MOR', 'SPT', 'MWKR'))]
    held = ['SW', 'LSW']
    M = np.full((len(rows), len(held)), np.nan)
    for c, reg in enumerate(held):
        for r, (label, base, pdrs) in enumerate(rows):
            if pdrs:
                vals = [_load(pp, reg) for pp in pdrs]
                vals = [v.mean() for v in vals if v is not None]
                M[r, c] = min(vals) if vals else np.nan
            else:
                v = _pooled(base.replace('{R}', reg), reg)
                M[r, c] = v.mean() if v is not None else np.nan
    if np.isnan(M).any():
        _stale('f23_parity_zeroshot', 'a method has no result on SW/LSW')
        return
    shade = np.zeros_like(M)
    for c in range(M.shape[1]):
        col = M[:, c]
        lo_, hi_ = np.nanmin(col), np.nanmax(col)
        shade[:, c] = (col - lo_) / (hi_ - lo_ + 1e-9)

    # Sized to print at 0.90\textwidth (6.44 in): the canvas IS the print size,
    # so every font below lands on paper at its stated point size.
    # Full text width (7.04 in = \textwidth - 9pt keyline padding) so the frame
    # spans the column band 1:1; height held so widening does not overflow.
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(7.04, 1.34), dpi=300,
        gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.50})

    y = np.arange(len(regimes))[::-1]
    ax.axvspan(-margin, margin, color=C_SPEC, alpha=0.16, lw=0,
               label=f'$\\pm${margin:g}% margin')
    for xv in (-margin, margin):
        ax.axvline(xv, color=C_SPEC, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axvline(0, color='0.35', lw=0.6, zorder=1)
    seen = set()
    for yi, lo, hi, m in zip(y, los, his, mids):
        v = _verdict(lo, hi)
        c, mk = STYLE[v]
        ax.plot([lo, hi], [yi, yi], lw=1.4, color=c, solid_capstyle='round', zorder=3)
        ax.plot([m], [yi], mk, ms=3.4, color=c, zorder=4,
                label=v if v not in seen else None)
        seen.add(v)
    ax.set_yticks(y); ax.set_yticklabels(regimes)
    ax.set_ylim(-0.7, len(regimes) - 0.3)
    # Left headroom so the legend sits in empty space: every CI starts at >= -0.3,
    # so the band left of -0.4 carries no data for the legend to cover.
    ax.set_xlim(-1.5, 1.9)
    ax.set_xlabel('paired makespan gap, joint $-$ specialist (%)')
    ax.set_ylabel('training regime')
    ax.grid(axis='x', lw=0.3, alpha=0.4); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    # Small legend tucked into the empty lower-left (all markers sit at x>=-0.3).
    leg = ax.legend(loc='lower left', ncol=1, handlelength=0.9,
                    handletextpad=0.4, fontsize=6.3, labelspacing=0.2,
                    borderpad=0.3, framealpha=0.92, edgecolor='0.75')
    leg.get_frame().set_linewidth(0.4)
    ax.set_title('(a) Specialist Parity (E1)', fontsize=9, pad=13)

    _cells(bx, shade, 'Blues', 0, 1.15)
    for c in range(M.shape[1]):
        best = int(np.nanargmin(M[:, c]))
        for r in range(M.shape[0]):
            bx.text(c, r, f'{M[r, c]:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold' if r == best else 'normal',
                    color='white' if shade[r, c] > 0.62 else '0.15')
    bx.set_xticks(range(len(held)))
    bx.set_xticklabels([f'{h}\n(held out)' for h in held])
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows])
    bx.tick_params(length=0)
    for sp_ in bx.spines.values():
        sp_.set_visible(False)
    bx.set_title('(b) Compositional Zero-Shot (E2): Mean Makespan (h)',
                 fontsize=9, pad=13)

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f23_parity_zeroshot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F23 written')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only = None
    for i, a in enumerate(_ARGV):
        if a == '--only':
            only = _ARGV[i + 1]
    if only in (None, 'f2'):
        f2_parity()
    if only in (None, 'f3'):
        f3_zeroshot()
    if only in (None, 'f23'):
        f23_parity_zeroshot()
    if only in (None, 'f4'):
        f4_causality()
    if only in (None, 'f5'):
        f5_anytime()
