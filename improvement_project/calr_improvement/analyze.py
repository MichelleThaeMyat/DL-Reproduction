"""Summarize all runs and draw figures without selecting methods by test scores."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results/controlled');a=ap.parse_args();p=Path(a.results)
    runs=json.loads((p/'runs.json').read_text());methods=list(dict.fromkeys(r['method'] for r in runs));summary=[]
    for method in methods:
        rs=[r for r in runs if r['method']==method];row={'method':method,'n':len(rs)}
        for metric in ['test_accuracy','test_loss','seconds','selected_epoch']:
            values=np.array([r[metric] for r in rs]);row[metric+'_mean']=float(values.mean());row[metric+'_sd']=float(values.std(ddof=1))
        summary.append(row)
    with (p/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    base={r['seed']:r for r in runs if r['method']=='CALR'};paired={}
    for method in methods[1:]:
        rs=[r for r in runs if r['method']==method];d=np.array([r['test_accuracy']-base[r['seed']]['test_accuracy'] for r in rs]);l=np.array([r['test_loss']-base[r['seed']]['test_loss'] for r in rs])
        # Student t critical value for df=2; descriptive interval for this fixed split.
        ci=4.302652729911275*d.std(ddof=1)/np.sqrt(len(d)) if len(d)==3 else None
        paired[method]=dict(accuracy_differences_pp=d.tolist(),mean_accuracy_difference_pp=float(d.mean()),accuracy_sd_pp=float(d.std(ddof=1)),ci95_low=float(d.mean()-ci) if ci else None,ci95_high=float(d.mean()+ci) if ci else None,loss_differences=l.tolist(),mean_loss_difference=float(l.mean()))
    (p/'analysis.json').write_text(json.dumps(dict(summary=summary,paired=paired),indent=2))
    draw_figures(p, runs, methods, base)
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))

def draw_figures(p, runs, methods, base):
    """Readable figures with full-range loss, late detail and discrete seed pairs."""
    plt.rcParams.update({
        'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })
    styles = {
        'CALR': ('#5B9BD5', '-', 'o', 'CALR'),
        'EMA-CALR': ('#E5A35A', '--', 's', 'EMA'),
        'Cosine-CALR': ('#65B6A4', '-.', '^', 'Cosine'),
        'EMA-Cosine-CALR': ('#AD83C4', ':', 'D', 'EMA+cosine'),
        'SGD': ('#999999', '--', 'v', 'SGD'),
    }
    # Full progress and late detail use linear axes and the same measured data.
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    fig.subplots_adjust(left=.14, right=.98, top=.82, bottom=.11, hspace=.43)
    late_bounds = []
    for i, method in enumerate(methods):
        color, line, marker, label = styles[method]
        histories = np.array([[x['val_loss'] for x in r['history']]
                              for r in runs if r['method'] == method])
        mean, sd = histories.mean(0), histories.std(0, ddof=1)
        xx = np.arange(1, len(mean) + 1)
        late_start = max(1, len(mean)//2)
        for ax in axes:
            ax.plot(xx, mean, label=label, color=color, ls=line, marker=marker,
                    markevery=(i % 4, 5), markersize=3.4, markerfacecolor='white',
                    markeredgewidth=.75, lw=1.1, zorder=3)
            ax.fill_between(xx, mean-sd, mean+sd, color=color, alpha=.09, lw=0, zorder=1)
        late = xx >= late_start
        late_bounds.extend((mean-sd)[late].tolist())
        late_bounds.extend((mean+sd)[late].tolist())
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(1, len(mean))
    axes[0].set_xticks([1, 10, 20, len(mean)])
    axes[0].set_title('(a) Full training range', loc='left', fontsize=11, pad=5)
    axes[1].set_title('(b) Late-training detail · enlarged scale', loc='left', fontsize=11, pad=5)
    lo, hi = min(late_bounds), max(late_bounds)
    margin = max((hi-lo)*.12, .002)
    axes[1].set_ylim(lo-margin, hi+margin)
    axes[1].set_xlim(late_start, len(mean))
    axes[1].set_xticks([late_start, 20, 25, len(mean)])
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    axes[1].yaxis.set_major_locator(MultipleLocator(.01))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for ax in axes:
        ax.grid(axis='y', color='#E6E6E6', lw=.6, zorder=0)
        ax.tick_params(labelsize=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    fig.text(.025, .46, 'Validation cross-entropy', rotation=90, va='center', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55,1),
               ncol=5, frameon=False, fontsize=9, columnspacing=1.2, handlelength=2)
    fig.text(.55, .905, 'Lines: mean across 3 seeds   Shading: ±1 sample SD',
             ha='center', fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(p/f'validation_loss.{ext}', dpi=450)
    plt.close(fig)

    # Seeds are categorical replicates, so avoid connecting them with trend lines.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig.subplots_adjust(left=.28, right=.97, top=.84, bottom=.19)
    seeds = sorted(base)
    markers = ['o', 's', '^', 'D', 'v']
    variants = [m for m in methods if m != 'CALR']
    for i, method in enumerate(variants):
        color = styles[method][0]
        records = {r['seed']: r for r in runs if r['method'] == method}
        for j, seed in enumerate(seeds):
            y = i + (j-(len(seeds)-1)/2)*.23
            delta = records[seed]['test_accuracy'] - base[seed]['test_accuracy']
            ax.scatter(delta, y, marker=markers[j % len(markers)], s=55,
                       color=color, edgecolor='white', linewidth=.6, zorder=3)
            ax.annotate(f'{delta:+.2f}', (delta, y), xytext=(7,0),
                        textcoords='offset points', va='center', fontsize=11,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.3, alpha=.9))
    ax.axvline(0, color='#333333', lw=1, ls='--')
    ax.set_yticks(range(len(variants)), variants, fontsize=12)
    ax.set_ylim(len(variants)-.5, -.5)
    values=[r['test_accuracy']-base[r['seed']]['test_accuracy']
            for r in runs if r['method'] != 'CALR']
    ax.set_xlim(min(min(values)-.015,-.015), max(max(values)+.028,.028))
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Accuracy difference from CALR\n(percentage points)', fontsize=13)
    ax.grid(axis='x', color='#D9DFE5', lw=.7)
    from matplotlib.lines import Line2D
    handles=[Line2D([], [], color='#444444', marker=markers[j % len(markers)],
                    linestyle='None', markersize=6, label=f'Seed {seed}')
             for j,seed in enumerate(seeds)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.59,.99),
               ncol=len(seeds), frameon=False, fontsize=12, columnspacing=1)
    for ext in ['png', 'pdf']:
        fig.savefig(p/f'paired_accuracy.{ext}', dpi=450)
    plt.close(fig)

if __name__=='__main__':main()
