import argparse
import os
import subprocess

from utils.gzip_check import check_and_fix_gzip as check
from utils.separate_if_sacrum import separate 
from dice_score.ds_ts import compute_dice_per_label
from utils.mha2nifti import mha_to_nifti as m2n

def run_total_segmentator(input_img, output_folder):
    print("Running TotalSegmentator...")
    cmd = [
        "TotalSegmentator",
        "-i", input_img,
        "-o", output_folder,
        "--task", "total_mr",
        "--roi_subset", "intervertebral_discs"
    ]
    subprocess.run(cmd, check=True)
    print("TotalSegmentator finished.")
    
def main(args):
    seg_output_folder = os.path.join(args.work_dir, "segmentation_output")
    os.makedirs(seg_output_folder, exist_ok=True)
    
    #check if mha
    if args.image.endswith(".mha"):
        # Convert .mha to .nii.gz
        m2n(args.image, seg_output_folder)
        args.image = os.path.join(seg_output_folder, os.path.basename(args.image).replace(".mha", ".nii.gz"))
    
    #check if the gzip is actually a gzip file, if not fix it.
    check(args.image)

    # Step 1: TotalSegmentator
    run_total_segmentator(args.image, seg_output_folder)
    disc_mask_path = os.path.join(seg_output_folder, "intervertebral_discs.nii.gz")
    labeled_discs_path = os.path.join(args.work_dir, "labeled_discs.nii.gz")

    # Step 2: Label discs
    disc_labels = ["L5-Sacrum", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12", "T10-T11", "T9-T10", "T8-T9", "T7-T8", "T6-T7", "T5-T6", "T4-T5", "T3-T4", "T2-T3", "T1-T2"]

    separate(disc_mask_path, labeled_discs_path, label_txt_path, disc_labels=disc_labels)

    output_csv = os.path.join(args.work_dir, "dice_scores.csv")
     
     # Define the mapping of labels to their corresponding values
     ## Specific for the OSF dataset. Change if needed.
    label_mapping = {
        "BS_L1_2": 5,
        "BS_L2_3": 4,
        "BS_L3_4": 3,
        "BS_L4_5": 2
    }
    # Step 3: DICE computation
    compute_dice_per_label( args.ground_truth, labeled_discs_path, label_mapping, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline runner with internal imports")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()
    label_txt_path = os.path.join(args.work_dir, "label_txt_path_labels.txt")
    
    main(args)