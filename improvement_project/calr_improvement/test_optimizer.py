import math,unittest,torch
from .optimizers import SmoothedCALR,cosine_lr
from .reference_optimizers import GradientCALR
class Tests(unittest.TestCase):
    def test_scalar(self):
        p=torch.tensor([1.],dtype=torch.float64,requires_grad=True);o=SmoothedCALR([p]);v=1.;m=q=0.
        for t,g in enumerate([2.,-1.,0.,3.],1):
            p.grad=torch.tensor([g],dtype=torch.float64);o.step();m=.9*m+.1*g;q=.9*q+.1*abs(g)
            v-=.01*(m/(1-.9**t))/(math.sqrt(q/(1-.9**t)+.1)+1e-8)
            self.assertAlmostEqual(p.item(),v,places=12)
    def test_reference_equivalence(self):
        p=torch.tensor([1.,-2.],requires_grad=True);r=p.detach().clone().requires_grad_();a=SmoothedCALR([p],rho=0);b=GradientCALR([r])
        for g in [torch.tensor([.3,-.2]),torch.tensor([-.1,.4])]:
            p.grad=g;r.grad=g;a.step();b.step()
        torch.testing.assert_close(p,r,rtol=0,atol=0)
    def test_schedule(self):
        a=[cosine_lr(e,30) for e in range(30)];self.assertAlmostEqual(a[0],.01);self.assertAlmostEqual(a[-1],.001);self.assertTrue(all(x>=y for x,y in zip(a,a[1:])))
if __name__=='__main__':unittest.main()
