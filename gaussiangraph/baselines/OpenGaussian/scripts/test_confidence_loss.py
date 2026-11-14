import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.confidence_loss import confidence_losses
class ConfidenceLossTest(unittest.TestCase):
 def test_formula_and_zero_confidence(self):
  f=torch.tensor([[[0.,1.]]],requires_grad=True);m=torch.tensor([[[True,False]],[[False,True]]]);means=torch.tensor([[0.],[1.]],requires_grad=True)
  ls,lc=confidence_losses(f,m,means,torch.tensor([1.,0.]));torch.testing.assert_close(ls,torch.tensor(0.));torch.testing.assert_close(lc,torch.tensor(1/6))
  ls,lc=confidence_losses(f,m,means,torch.zeros(2));(ls+lc).backward();torch.testing.assert_close(f.grad,torch.zeros_like(f));torch.testing.assert_close(means.grad,torch.zeros_like(means))
 def test_alignment_and_empty_masks(self):
  f=torch.ones(2,3,4,requires_grad=True);m=torch.zeros(2,3,4,dtype=torch.bool);means=torch.zeros(2,2)
  with self.assertRaises(ValueError):confidence_losses(f,m,means,torch.ones(3))
  a,b=confidence_losses(f,m,means,torch.ones(2));self.assertEqual((a+b).item(),0)
if __name__=='__main__':unittest.main()
