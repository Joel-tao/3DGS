import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scene.control_follow import CFConfig,ControlFollow
class FeatureSpaceTest(unittest.TestCase):
 def test_unit_distance_invariant_to_individual_positive_scaling(self):
  torch.manual_seed(0);xyz=torch.randn(40,3);features=torch.randn(40,6)
  cf=ControlFollow(CFConfig(cluster_space='unit'))
  torch.testing.assert_close(cf.vectors(xyz,features),cf.vectors(xyz,features*torch.linspace(.1,10,40)[:,None]),atol=1e-6,rtol=1e-5)
 def test_legacy_default_unchanged(self):
  xyz=torch.randn(4,3);features=torch.randn(4,6)
  torch.testing.assert_close(ControlFollow().vectors(xyz,features),torch.cat([features,xyz*.5],dim=1))
if __name__=='__main__':unittest.main()
