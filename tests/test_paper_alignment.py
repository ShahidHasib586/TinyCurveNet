import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
sys.path.insert(0, str(ROOT / "src"))
from model import TinyCurveCCMNet
from losses import TinyCurveLoss
from dataset import PairedImageFolder
from PIL import Image
import numpy as np
import tempfile

class PaperAlignmentTests(unittest.TestCase):
    def test_paired_crop_and_flip_stay_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"low", Path(tmp)/"high"
            a.mkdir(); b.mkdir()
            arr = np.arange(61*77*3, dtype=np.uint8).reshape(61,77,3)
            Image.fromarray(arr).save(a/"pair.png")
            Image.fromarray(arr).save(b/"pair.png")
            ds = PairedImageFolder(a,b,img_size=32,train=True)
            for _ in range(20):
                x,y,_ = ds[0]
                self.assertEqual(x.shape, (3,32,32))
                self.assertTrue(torch.equal(x,y))
            small = PairedImageFolder(a,b,img_size=96,train=True)
            x,y,_ = small[0]
            self.assertTrue(torch.equal(x,y))
            self.assertEqual(x.shape, (3,96,96))

    def test_shape_range_and_existing_checkpoint(self):
        model = TinyCurveCCMNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),11642)
        ck = torch.load(ROOT/"checkpoints/stage_05_best.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(ck["model"],strict=True)
        with torch.no_grad():
            y,g,gamma,m,b = model(torch.rand(1,3,33,49))
        self.assertEqual(y.shape,(1,3,33,49))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(((y>=0)&(y<=1)).all())
        self.assertTrue(((g>=.6)&(g<=2.)).all())
        self.assertTrue(((gamma>=.6)&(gamma<=2.2)).all())

    def test_ssim_contributes_to_gradient(self):
        x=torch.rand(1,3,32,32,requires_grad=True); y=torch.rand_like(x)
        loss,_=TinyCurveLoss(w_l1=0,w_ssim=1,w_lpips=0,use_lpips=False)(x,y)
        loss.backward()
        self.assertGreater(x.grad.abs().sum().item(),0)

    def test_pure_objectives(self):
        x=torch.rand(1,3,32,32,requires_grad=True); y=torch.rand_like(x)
        for objective,expected in [("l1",(x-y).abs().mean()),("mse",(x-y).square().mean())]:
            fn=TinyCurveLoss(objective=objective)
            actual,_=fn(x,y)
            torch.testing.assert_close(actual,expected)
            self.assertIsNone(fn.lpips_fn)

if __name__ == "__main__":
    unittest.main()
