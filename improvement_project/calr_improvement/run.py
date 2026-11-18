"""Matched training; test evaluation occurs only after all validation selection."""
import argparse,copy,csv,hashlib,json,os,platform,time
from pathlib import Path
import torch,torchvision
from torchvision.datasets import MNIST
from torch import nn
from .optimizers import SmoothedCALR,cosine_lr
from .reference_optimizers import GradientCALR
METHODS=['CALR','EMA-CALR','Cosine-CALR','EMA-Cosine-CALR','SGD']
def model():return nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
def normalize(d):return d.data.reshape(-1,784).float().div(255).sub(.5).div(.5),d.targets
@torch.no_grad()
def evaluate(m,x,y):
    m.eval();loss=correct=0
    for start in range(0,len(y),1000):
        p=m(x[start:start+1000]);z=y[start:start+1000];loss+=nn.functional.cross_entropy(p,z,reduction='sum').item();correct+=p.argmax(1).eq(z).sum().item()
    return dict(loss=loss/len(y),accuracy=100*correct/len(y))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=os.environ.get('CALR_DATA_DIR','data'));ap.add_argument('--output',default='results/controlled');ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);a=ap.parse_args()
    if a.epochs<2:ap.error('epochs must be at least two')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Choose a new empty output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    config=dict(args=vars(a),methods=METHODS,split_seed=12345,train=50000,validation=10000,test=10000,batch_size=64,initial_lr=.01,minimum_lr=.001,rho=.9,alpha=.1,beta1=.9,eps=1e-8,selection='minimum validation cross entropy; earliest tie',primary='paired test accuracy difference EMA-Cosine-CALR minus CALR',python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),threads=1,source_sha256=hashes,status='running')
    (out/'config.json').write_text(json.dumps(config,indent=2));started=time.perf_counter()
    x,y=normalize(MNIST(a.data,train=True,download=True));ids=torch.randperm(len(y),generator=torch.Generator().manual_seed(12345));tr,va=ids[:50000],ids[50000:];torch.save(dict(train=tr,validation=va),out/'split.pt');xt,yt=x[tr],y[tr];xv,yv=x[va],y[va];runs=[]
    for seed in a.seeds:
        torch.manual_seed(seed);initial=copy.deepcopy(model().state_dict());ih=hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial.values())).hexdigest()
        for name in METHODS:
            torch.manual_seed(seed);m=model();m.load_state_dict(initial);ctor=SmoothedCALR if name.startswith('EMA') else GradientCALR
            opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9) if name=='SGD' else ctor(m.parameters(),lr=.01)
            order=torch.Generator().manual_seed(seed+10000);history=[];best=float('inf');selected=None;t0=time.perf_counter()
            for epoch in range(a.epochs):
                lr=cosine_lr(epoch,a.epochs) if 'Cosine' in name else .01
                for g in opt.param_groups:g['lr']=lr
                m.train();perm=torch.randperm(len(yt),generator=order);ls=0.
                for start in range(0,len(yt),64):
                    ix=perm[start:start+64];opt.zero_grad(set_to_none=True);pred=m(xt[ix]);loss=nn.functional.cross_entropy(pred,yt[ix]);loss.backward();opt.step();ls+=loss.item()*len(ix)
                val=evaluate(m,xv,yv);history.append(dict(epoch=epoch+1,lr=lr,train_loss=ls/len(yt),val_loss=val['loss'],val_accuracy=val['accuracy']))
                if val['loss']<best:best=val['loss'];selected=dict(epoch=epoch+1,state=copy.deepcopy(m.state_dict()),validation=val)
                if (epoch+1)%10==0:print(seed,name,epoch+1,round(val['loss'],5),round(val['accuracy'],2),flush=True)
            stem=f'{name}_seed{seed}';torch.save(selected['state'],out/'checkpoints'/f'{stem}.pt');r=dict(method=name,seed=seed,initial_sha256=ih,order_seed=seed+10000,selected_epoch=selected['epoch'],val_loss=best,val_accuracy=selected['validation']['accuracy'],seconds=time.perf_counter()-t0,history=history);(out/f'{stem}.json').write_text(json.dumps(r,indent=2));runs.append(r)
    xx,yy=normalize(MNIST(a.data,train=False,download=True))
    for r in runs:
        m=model();m.load_state_dict(torch.load(out/'checkpoints'/f"{r['method']}_seed{r['seed']}.pt",weights_only=True));v=evaluate(m,xx,yy);r.update(test_loss=v['loss'],test_accuracy=v['accuracy'])
    (out/'runs.json').write_text(json.dumps(runs,indent=2));keys=['method','seed','selected_epoch','val_loss','val_accuracy','test_loss','test_accuracy','seconds']
    with (out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in runs)
    config.update(status='completed',total_seconds=time.perf_counter()-started);(out/'config.json').write_text(json.dumps(config,indent=2));print('COMPLETE',config['total_seconds'],flush=True)
if __name__=='__main__':main()
