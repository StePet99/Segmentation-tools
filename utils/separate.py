import SimpleITK as sitk
import numpy as np
import os
import argparse

# Expected vertebrae anatomical order
vertebrae_order = [
    "vertebrae_T12", "vertebrae_L1", "vertebrae_L2",
    "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "sacrum"
]

def load_mask_and_center(path):
    """Load binary mask and compute the center of the foreground voxels."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        raise ValueError(f"No foreground in mask: {path}")
    center_voxel = np.mean(coords, axis=0)
    return img, center_voxel[::-1]  # from z,y,x to x,y,z

def main(vertebrae_folder, disc_path, output_path, label_txt_path):
    vertebrae_centers = []
    ref_img = None

    # Load vertebrae masks and compute centers
    for name in vertebrae_order:
        path = os.path.join(vertebrae_folder, f"{name}.nii.gz")
        if not os.path.exists(path):
            print(f"Skipping missing vertebra: {name}")
            continue
        img, center = load_mask_and_center(path)
        vertebrae_centers.append((name, center))
        if ref_img is None:
            ref_img = img

    # Build dictionary of vertebra name to center
    vertebrae_name_to_center = dict(vertebrae_centers)
    present_vertebrae = set(vertebrae_name_to_center.keys())

    # Build list of valid consecutive vertebrae pairs
    valid_pairs = []
    for i in range(len(vertebrae_order) - 1):
        v1, v2 = vertebrae_order[i], vertebrae_order[i + 1]
        if v1 in present_vertebrae and v2 in present_vertebrae:
            c1 = vertebrae_name_to_center[v1]
            c2 = vertebrae_name_to_center[v2]
            valid_pairs.append((v1, c1[2], v2, c2[2]))  # z coordinates

    # Load disc mask and compute connected components
    disc_img = sitk.ReadImage(disc_path)
    disc_arr = sitk.GetArrayFromImage(disc_img)
    output_arr = np.zeros_like(disc_arr, dtype=np.uint8)

    binary_mask = (disc_arr > 0).astype(np.uint8)
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask))
    cc_arr = sitk.GetArrayFromImage(cc)
    num_discs = int(cc_arr.max())

    label_dict = {}
    label = 1

    for i in range(1, num_discs + 1):
        coords = np.argwhere(cc_arr == i)
        if coords.size == 0:
            continue
        center = np.mean(coords, axis=0)[::-1]  # x, y, z
        center_z = center[2]

        # Find the correct vertebra pair based on z coordinate and assign label
        for v1, z1, v2, z2 in valid_pairs:
            if z1 <= center_z <= z2 or z2 <= center_z <= z1:
                output_arr[cc_arr == i] = label
                label_dict[label] = f"{v1.replace('vertebrae_', '')}-{v2.replace('vertebrae_', '')}"
                label += 1
                break

    # Save output labeled disc mask
    output_img = sitk.GetImageFromArray(output_arr)
    output_img.CopyInformation(disc_img)
    sitk.WriteImage(output_img, output_path)
    print(f"Saved labeled discs to: {output_path}")

    # Save label-to-name mapping
    with open(label_txt_path, 'w') as f:
        for k, v in label_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved label map to: {label_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign labels to intervertebral discs based on vertebra masks.")
    parser.add_argument("--vertebrae_folder", type=str, required=True, help="Folder containing vertebra masks (.nii.gz)")
    parser.add_argument("--disc_path", type=str, required=True, help="Path to the single disc mask (.nii.gz)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the labeled disc mask (.nii.gz)")
    args = parser.parse_args()

    # Generate label text path from output path
    label_txt_path = os.path.splitext(args.output_path)[0] + "_labels.txt"

    main(args.vertebrae_folder, args.disc_path, args.output_path, label_txt_path)
