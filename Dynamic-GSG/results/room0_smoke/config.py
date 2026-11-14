"""8 GB GPU smoke run, not the paper benchmark configuration."""
import os
from copy import deepcopy
from runpy import run_path

config = deepcopy(run_path("configs/replica/dgsg.py")["config"])
config.update(workdir="/media/joe/Data/Dynamic-GSG/experiments", run_name=os.getenv("DGSG_RUN_NAME", "room0_smoke"), save_checkpoints=True)
config["data"].update(basedir="/media/joe/Data/SplaTAM/data/Replica", desired_image_height=240, desired_image_width=424, num_frames=int(os.getenv("DGSG_NUM_FRAMES", "10")))
config["tracking"].update(use_gt_poses=True, modify_real_gt_poses=False)
config["mapping"]["num_iters"] = int(os.getenv("DGSG_MAPPING_ITERS", "20"))
config["lang"].update(enable_object_captions=False, clip_model_name="ViT-B-32", clip_pretrained="/media/joe/Data/Dynamic-GSG/models/clip_vit_b32_laion2b.bin", sys_prompt_file="./configs/prompts/parsing_query.txt", obj_prompt_file="./configs/prompts/parsing_objects.txt", obj_caption_file="./configs/prompts/parsing_objects_caption.txt")
for key, filename in [("variables_path", "variables.npz"), ("keyframe_list_path", "keyframelist.pkl.gz")]:
    config["viz"][key] = os.path.join(config["workdir"], config["run_name"], filename)

config["viz"].update(clip_model_name=config["lang"]["clip_model_name"], clip_pretrained=config["lang"]["clip_pretrained"])
