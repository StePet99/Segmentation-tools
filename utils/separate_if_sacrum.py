import SimpleITK as sitk
import numpy as np
import argparse
import os

# Expected disc label names from bottom to top
disc_labels = ["L5-Sacrum", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12", "T10-T11", "T9-T10", "T8-T9", "T7-T8", "T6-T7", "T5-T6", "T4-T5", "T3-T4", "T2-T3", "T1-T2"]

def separate(disc_path, output_path, label_txt_path, disc_labels=disc_labels):
    # Load disc mask
    disc_img = sitk.ReadImage(disc_path)
    disc_arr = sitk.GetArrayFromImage(disc_img)

    # Create binary mask
    binary_mask = (disc_arr > 0).astype(np.uint8)

    # Connected component labeling
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask))
    cc_arr = sitk.GetArrayFromImage(cc)
    num_discs = int(cc_arr.max())

    if num_discs != len(disc_labels):
        print(f"Found {num_discs} discs.")

    # Compute center Z for each component
    disc_centers = []
    for i in range(1, num_discs + 1):
        coords = np.argwhere(cc_arr == i)
        if coords.size == 0:
            continue
        center = np.mean(coords, axis=1)[::-1]  # x, y, z
        disc_centers.append((i, center[2]))     # label index, z

    # Sort discs from bottom (highest z) to top (lowest z)
    disc_centers.sort(key=lambda z: z[1])

    # Assign new labels in anatomical order
    output_arr = np.zeros_like(cc_arr, dtype=np.uint8)
    label_dict = {}

    for new_label, (old_label, _) in enumerate(disc_centers, start=1):
        output_arr[cc_arr == old_label] = new_label
        if new_label <= len(disc_labels):
            label_dict[new_label] = disc_labels[new_label - 1]
        else:
            label_dict[new_label] = f"Unknown_{new_label}"

    # Save output mask
    output_img = sitk.GetImageFromArray(output_arr)
    output_img.CopyInformation(disc_img)
    sitk.WriteImage(output_img, output_path)
    print(f"Saved labeled discs to: {output_path}")

    # Save label dictionary
    with open(label_txt_path, 'w') as f:
        for k, v in label_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved label map to: {label_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label intervertebral discs from bottom to top assuming full lumbar column.")
    parser.add_argument("--disc_path", type=str, required=True, help="Path to the disc mask (single .nii.gz)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the labeled disc mask")
    args = parser.parse_args()

    label_txt_path = os.path.splitext(args.output_path)[0] + "_labels.txt"

    separate(args.disc_path, args.output_path, label_txt_path)
