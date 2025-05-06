import os
import nibabel as nib
import numpy as np
import argparse
import csv
from tqdm import tqdm

def dice_score(y_true, y_pred):
    """
    Compute the Dice score between two binary arrays.
    """
    intersection = np.sum((y_true > 0) & (y_pred > 0))
    size_true = np.sum(y_true > 0)
    size_pred = np.sum(y_pred > 0)

    if size_true + size_pred == 0:
        return 1.0  # 1 if both are empty
    else:
        return (2. * intersection) / (size_true + size_pred)

def extract_label(mask, label_value):
    """
    Extract only the voxels corresponding to a specific label.
    """
    return (mask == label_value).astype(np.uint8)

def compute_dice_per_label(gt_folder, pred_file, label_mapping, output_csv):
    """
    For each GT file:
    - Extract the corresponding label from the predicted segmentation
    - Compute the Dice score
    - Save all results to a CSV file
    """
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".nii.gz")])

    if len(gt_files) == 0:
        raise ValueError("Ground Truth folder is empty or does not contain any '.nii.gz' files.")

    pred_img = nib.load(pred_file)
    pred_data = pred_img.get_fdata()

    dice_scores = []

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File", "Label", "Dice Score"])

        for gt_file in tqdm(gt_files, desc="Computing Dice", unit="file"):
            gt_img = nib.load(os.path.join(gt_folder, gt_file))
            gt_data = gt_img.get_fdata()

            # Get the label associated with the GT file
            label_name = os.path.splitext(os.path.splitext(gt_file)[0])[0]  # removes .nii.gz
            label_value = label_mapping.get(label_name, None)

            if label_value is None:
                print(f"No label found for '{label_name}', skipping.")
                continue

            pred_label = extract_label(pred_data, label_value)

            dice = dice_score(gt_data, pred_label)
            dice_scores.append(dice)

            print(f"{gt_file} (Label {label_value}): Dice = {dice:.4f}")
            writer.writerow([gt_file, label_value, f"{dice:.4f}"])

    if dice_scores:
        avg_dice = np.mean(dice_scores)
        print("\n Average Dice Score over all files:", f"{avg_dice:.4f}")
        with open(output_csv, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([])
            writer.writerow(["Average Dice", "", f"{avg_dice:.4f}"])
    else:
        print("\n No Dice Scores were computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Dice Scores between multiple GT files and a global predicted segmentation file.")
    parser.add_argument("-gt", "--ground_truth_folder", required=True, help="Folder containing GT segmentations (.nii.gz)")
    parser.add_argument("-p", "--prediction_file", required=True, help="Global predicted segmentation file (.nii.gz)")
    parser.add_argument("-o", "--output_csv", required=True, help="Path to save the output CSV file")

    args = parser.parse_args()

    # Define the mapping of labels to their corresponding values
    label_mapping = {
        "BS_L1_2": 4,
        "BS_L2_3": 3,
        "BS_L3_4": 2,
        "BS_L4_5": 1
        # Modify dictionary based on dataset
    }

    compute_dice_per_label(args.ground_truth_folder, args.prediction_file, label_mapping, args.output_csv)
