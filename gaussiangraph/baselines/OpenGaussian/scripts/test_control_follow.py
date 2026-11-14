import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scene.control_follow import CFConfig,ControlFollow,GradientHistory
class CFTest(unittest.TestCase):
 def test_online_follow_updates_center(self):
  xyz=torch.tensor([[0.,0,0],[.15,0,0],[.25,0,0],[5.,0,0]])
  features=torch.zeros(4,6);cf=ControlFollow(CFConfig(position_scale=1,control_threshold=.1,follow_threshold=.2))
  cf.controls=torch.tensor([0]);cf.control(xyz,features);cf.follow(xyz,features)
  self.assertEqual(cf.labels.tolist(),[0,0,0,1])
 def test_quantization_values_and_gradient(self):
  x=torch.tensor([[1.,2.],[3.,4.],[8.,9.]],requires_grad=True);cf=ControlFollow();cf.labels=torch.tensor([0,0,-1])
  q=cf.quantize(x);torch.testing.assert_close(q,torch.tensor([[2.,3.],[2.,3.],[8.,9.]]))
  q.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))
  with self.assertRaises(ValueError):cf.quantize(torch.zeros(4,2))
 def test_gradient_history_remap(self):
  h=GradientHistory(3,'cpu');h.update(torch.tensor([[.01,0],[2.,0],[0.,0]]),torch.tensor([True,True,False]))
  self.assertEqual(h.stable(.1).tolist(),[True,False,False]);h.remap(torch.tensor([1,0,-1]))
  self.assertEqual(h.stable(.1).tolist(),[False,True,False])
 def test_refine_separates_features(self):
  f=torch.cat((torch.zeros(5,6),torch.ones(5,6)));cf=ControlFollow(CFConfig(refine_threshold=.1));cf.labels=torch.zeros(10,dtype=torch.long);cf.refine(f)
  self.assertEqual(len(cf.labels.unique()),2)
 def test_fpfh_returns_original_stable_indices(self):
  torch.manual_seed(1);xyz=torch.randn(100,3);stable=torch.arange(100)%2==0
  cf=ControlFollow(CFConfig(control_points=5000,voxel_size=.01,radius=1,max_nn=20));ids=cf.select_controls(xyz,stable)
  self.assertEqual(len(ids),len(ids.unique()));self.assertTrue(stable[ids].all());self.assertEqual(len(ids),50)
 def test_unit_refinement_is_invariant_to_feature_magnitude(self):
  f=torch.zeros(65,6);f[:30,0]=torch.logspace(-1,2,30);f[30:60,1]=torch.logspace(-1,2,30);f[60:,2]=1
  config=CFConfig(refine_space='unit',refine_min_points=10,refine_threshold=.1)
  cf=ControlFollow(config);cf.labels=torch.zeros(65,dtype=torch.long);cf.refine(f)
  self.assertEqual(len(cf.labels.unique()),2)
  self.assertTrue((cf.labels[:30]==cf.labels[0]).all());self.assertTrue((cf.labels[30:60]==cf.labels[30]).all())
  self.assertNotEqual(cf.labels[0],cf.labels[30]);self.assertEqual(len(cf.labels),65)
if __name__=='__main__':unittest.main()
