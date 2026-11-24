"""Post-run training-gradient diagnostic; no hyperparameter selection."""
import argparse,json
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from .run import model,normalize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
    p=a.results;c=json.loads((p/'config.json').read_text());torch.set_num_threads(1)
    x,y=normalize(MNIST(a.data or c['args']['data'],train=True,download=True));split=torch.load(p/'split.pt',weights_only=True);tr=split['train'];xx,yy=x[tr],y[tr];rows=[]
    for seed in c['args']['seeds']:
        ix=torch.randperm(len(yy),generator=torch.Generator().manual_seed(seed+10000))[:c['batch_size']]
        for stage in ['initial','selected_CALR']:
            torch.manual_seed(seed);m=model()
            if stage!='initial':m.load_state_dict(torch.load(p/'checkpoints'/f'CALR_seed{seed}.pt',weights_only=True))
            torch.nn.functional.cross_entropy(m(xx[ix]),yy[ix]).backward();g=torch.cat([z.grad.flatten().abs() for z in m.parameters()]);rows.append(dict(seed=seed,stage=stage,batch_size=c['batch_size'],median_abs_gradient=g.median().item(),mean_abs_gradient=g.mean().item(),fraction_below_alpha_over_10=(g<c['alpha']/10).float().mean().item()))
    (p/'gradient_diagnostic.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
