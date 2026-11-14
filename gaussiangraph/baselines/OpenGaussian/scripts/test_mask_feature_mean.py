"""Regression checks for mask statistics and their training gradients."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from utils.opengs_utlis import mask_feature_mean


def reference(features, masks, image):
    # Direct definition on small tensors, including upstream empty-mask behavior.
    expanded = masks[:, None].expand(-1, features.shape[0], -1, -1)
    weights = expanded if image is None else expanded * image.float()
    values = features[None] * weights
    count = weights.sum((2, 3)).clamp(min=1)
    mean = values.sum((2, 3)) / count
    variance = (((values - mean[:, :, None, None]) * expanded).square()
                .sum((2, 3)) / count).mean(1)
    return mean, variance, count[:, 0]


class MaskStatisticsTest(unittest.TestCase):
    def test_statistics_and_gradients(self):
        torch.manual_seed(12)
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            for masked in (False, True):
                x = torch.randn(6, 11, 13, device=device, requires_grad=True)
                masks = torch.rand(7, 11, 13, device=device) > .6
                masks[0] = False
                image = (torch.rand(1, 11, 13, device=device) > .3) if masked else None
                expected = reference(x, masks, image)
                actual = mask_feature_mean(x, masks, image, return_var=True)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                a_grad = torch.autograd.grad(actual[0].sum() + actual[1].sum(), x, retain_graph=True)[0]
                b_grad = torch.autograd.grad(expected[0].sum() + expected[1].sum(), x)[0]
                torch.testing.assert_close(a_grad, b_grad, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(mask_feature_mean(x, masks, image), expected[0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_original_resolution_memory(self):
        with torch.no_grad():
            features = torch.randn(6, 731, 988, device='cuda')
            masks = torch.rand(64, 731, 988, device='cuda') > .8
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            mask_feature_mean(features, masks, return_var=True)
            torch.cuda.synchronize()
            extra = torch.cuda.max_memory_allocated() - before
            self.assertLess(extra, 512 * 1024**2)


if __name__ == '__main__':
    unittest.main()
