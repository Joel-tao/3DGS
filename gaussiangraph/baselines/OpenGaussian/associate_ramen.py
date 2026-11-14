"""Resume only final language association from the saved full training checkpoint."""
from argparse import ArgumentParser
from pathlib import Path
import torch
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from train import construct_pseudo_ins_feat
from utils.opengs_utlis import load_code_book

parser = ArgumentParser()
model = ModelParams(parser)
pipeline = PipelineParams(parser)
parser.add_argument('--iteration', type=int, default=70000)
parser.add_argument('--root_node_num', type=int, default=64)
parser.add_argument('--leaf_node_num', type=int, default=10)
args = parser.parse_args()
with torch.no_grad():
    gaussian = GaussianModel(args.sh_degree)
    scene = Scene(model.extract(args), gaussian, load_iteration=args.iteration, shuffle=False)
    # The PLY stores quantized features. Restore the original continuous features
    # from the checkpoint for origin_feat=True, without allocating an optimizer.
    checkpoint = Path(args.model_path) / f'chkpnt{args.iteration}.pth'
    params, iteration = torch.load(checkpoint, map_location='cpu')
    if iteration != args.iteration or len(params) != 14:
        raise ValueError('Unexpected checkpoint format or iteration')
    gaussian.active_sh_degree = params[0]
    names = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
             '_opacity', '_ins_feat', '_ins_feat_q')
    for name, value in zip(names, params[1:9]):
        setattr(gaussian, name, torch.nn.Parameter(value.detach().cuda(), requires_grad=False))
    del params
    _, labels = load_code_book(str(Path(args.model_path) / 'point_cloud' /
                                  f'iteration_{args.iteration}' / 'leaf_code_book'))
    labels = torch.from_numpy(labels).cuda()
    if labels.numel() != gaussian._xyz.shape[0]:
        raise ValueError('Codebook/Gaussian count mismatch')
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    construct_pseudo_ins_feat(scene, render, (pipeline.extract(args), background, args.iteration),
                             cluster_indices=labels, mode='lang', root_num=args.root_node_num,
                             leaf_num=args.leaf_node_num, sam_level=3, save_memory=True)
    print('Language association complete.', flush=True)
