"""Check saved model scores and paired evaluation metadata."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from .run import model,normalize,evaluate,METHODS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
 p=a.results;cfg=json.loads((p/'config.json').read_text());runs=json.loads((p/'runs.json').read_text());torch.set_num_threads(1)
 assert cfg['status']=='completed' and len(runs)==len(METHODS)*len(cfg['args']['seeds'])
 core={'run.py','optimizers.py','reference_optimizers.py','__init__.py'}
 assert core <= cfg['source_sha256'].keys(), 'Missing training source hashes'
 changed=[]
 for name,digest in cfg['source_sha256'].items():
  matches=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
  if name in core:assert matches, f'Training source changed: {name}'
  elif not matches:changed.append(name)
 split=torch.load(p/'split.pt',weights_only=True)
 assert len(split['train'])==cfg['train'] and len(split['validation'])==cfg['validation']
 assert torch.equal(torch.cat([split['train'],split['validation']]).sort().values,torch.arange(60000)), 'Invalid split'
 assert {(r['method'],r['seed']) for r in runs}=={(m,s) for m in METHODS for s in cfg['args']['seeds']}
 for seed in cfg['args']['seeds']:
  same=[r for r in runs if r['seed']==seed];assert len(set(r['initial_sha256'] for r in same))==1;assert len(set(r['order_seed'] for r in same))==1
 x,y=normalize(MNIST(a.data or cfg['args']['data'],train=False,download=True))
 for r in runs:
  h=r['history'];assert len(h)==cfg['args']['epochs'];assert min(h,key=lambda v:v['val_loss'])['epoch']==r['selected_epoch']
  m=model();m.load_state_dict(torch.load(p/'checkpoints'/f"{r['method']}_seed{r['seed']}.pt",weights_only=True));v=evaluate(m,x,y)
  assert abs(v['loss']-r['test_loss'])<1e-12 and v['accuracy']==r['test_accuracy']
 report=dict(passed=True,runs=len(runs),changed_nontraining_sources=changed,checks=['training source hashes','complete disjoint train/validation split','paired initialization and shuffle seeds','validation checkpoint choice','saved checkpoint test metrics']);(p/'verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
