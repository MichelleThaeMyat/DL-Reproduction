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
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#17324D','#B55D22','#228064','#7944A3','#777777']
    fig,ax=plt.subplots(figsize=(3.5,2.35),layout='constrained')
    for method,color in zip(methods,colors):
        histories=np.array([[x['val_loss'] for x in r['history']] for r in runs if r['method']==method]);mean=histories.mean(0);sd=histories.std(0,ddof=1);xx=np.arange(1,len(mean)+1)
        ax.plot(xx,mean,label=method,color=color,lw=1.2);ax.fill_between(xx,mean-sd,mean+sd,color=color,alpha=.1)
    ax.set(xlabel='Epoch',ylabel='Validation cross-entropy',ylim=(.04,.24));ax.legend(fontsize=6.5,ncol=2);ax.grid(alpha=.2);fig.savefig(p/'validation_loss.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.5,2.35),layout='constrained')
    seeds=sorted(base)
    for method,color in zip(methods[1:],colors[1:]):
        vals=paired[method]['accuracy_differences_pp'];ax.plot(seeds,vals,'o-',label=method,color=color,lw=1)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Paired initialization seed',ylabel='Accuracy change vs CALR (pp)',xticks=seeds);ax.legend(fontsize=6.5,ncol=2);ax.grid(alpha=.2);fig.savefig(p/'paired_accuracy.png',dpi=300);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))
if __name__=='__main__':main()
