import os
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def evalute(gt_base, pred_base, scene_name):
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    frame_names = scene_gt_frames[scene_name]

    ious = []
    missing_predictions = 0
    for frame in frame_names:
        print("frame:", frame)
        gt_floder = os.path.join(gt_base, frame)
        file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg')]
        for file_name in file_names:
            base_name = os.path.splitext(file_name)[0]
            gt_obj_path = os.path.join(gt_floder, file_name)
            pred_obj_path = os.path.join(pred_base, frame + "_" + base_name + '.png')
            if not os.path.exists(pred_obj_path):
                missing_predictions += 1
                print(f"Missing pred file for {file_name}, skipping...")
                print(f"IoU for {file_name}: 0")
                ious.append(0.0)
                continue
            mask_gt = load_image_as_binary(gt_obj_path)
            mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
            iou = calculate_iou(mask_gt, mask_pred)
            ious.append(iou)
            print(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}")
    
    # Acc.
    total_count = len(ious)
    if total_count == 0:
        raise ValueError("No ground-truth masks found")
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    # mIoU
    average_iou = np.mean(ious)
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Acc@0.25: {count_iou_025/total_count:.4f}")
    print(f"Acc@0.5: {count_iou_05/total_count:.4f}")
    return {"scene": scene_name, "count": total_count, "missing_predictions": missing_predictions, "mIoU": float(average_iou),
            "Acc@0.25": float(count_iou_025/total_count),
            "Acc@0.5": float(count_iou_05/total_count)}

if __name__ == "__main__":
    parser = ArgumentParser("Compute LeRF IoU")
    parser.add_argument("--gt", required=True, help="Scene GT directory containing frame folders")
    parser.add_argument("--pred", required=True, help="renders_cluster_silhouette directory")
    parser.add_argument("--output", help="Optional metrics JSON file")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    args = parser.parse_args()
    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")

    # TODO: change
    path_gt = args.gt
    # renders_cluster_silhouette is the predicted mask
    path_pred = args.pred
    result = evalute(path_gt, path_pred, args.scene_name)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
