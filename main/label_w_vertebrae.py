import SimpleITK as sitk
import numpy as np
import os
import argparse

# Expected order of vertebrae
vertebrae_order = [
    "vertebrae_T12", "vertebrae_L1", "vertebrae_L2",
    "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "sacrum"
]

def load_mask_and_center(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        raise ValueError(f"No foreground in mask: {path}")
    center_voxel = np.mean(coords, axis=0)
    return img, center_voxel[::-1] 

def main(vertebrae_folder, disc_path, output_path, label_txt_path):
    vertebrae_centers = []
    ref_img = None

    # Upload vertebrae masks and find centers
    for name in vertebrae_order:
        path = os.path.join(vertebrae_folder, f"{name}.nii.gz")
        if not os.path.exists(path):
            print(f"Skipping missing vertebra: {name}")
            continue
        img, center = load_mask_and_center(path)
        vertebrae_centers.append((name, center))
        if ref_img is None:
            ref_img = img

    # Order vertebrae by z coordinate
    vertebrae_centers.sort(key=lambda x: x[1][2])

    # Upload disc mask
    disc_img = sitk.ReadImage(disc_path)
    disc_arr = sitk.GetArrayFromImage(disc_img)
    output_arr = np.zeros_like(disc_arr, dtype=np.uint8)

    # Create binary mask and connected components
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
        center = np.mean(coords, axis=0)[::-1]  # z,y,x → x,y,z
        center_z = center[2]

        # Find the vertebrae on top and bottom of the disc, assign label
        for v1, v2 in zip(vertebrae_centers[:-1], vertebrae_centers[1:]):
            z1 = v1[1][2]
            z2 = v2[1][2]
            if z1 <= center_z <= z2:
                output_arr[cc_arr == i] = label
                label_dict[label] = f"{v1[0].replace('vertebrae_', '')}-{v2[0].replace('vertebrae_', '')}"
                label += 1
                break

    # Save output mask
    output_img = sitk.GetImageFromArray(output_arr)
    output_img.CopyInformation(disc_img)
    sitk.WriteImage(output_img, output_path)
    print(f"Saved labeled discs to: {output_path}")

    # Save dictionary of labels
    with open(label_txt_path, 'w') as f:
        for k, v in label_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved label map to: {label_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assegna etichette ai dischi intervertebrali usando le maschere delle vertebre.")
    parser.add_argument("--vertebrae_folder", type=str, required=True, help="Cartella con le maschere delle vertebre (.nii.gz)")
    parser.add_argument("--disc_path", type=str, required=True, help="Percorso alla maschera dei dischi (unica .nii.gz)")
    parser.add_argument("--output_path", type=str, required=True, help="Percorso del file NIfTI con dischi etichettati")
    args = parser.parse_args()

    # Genera automaticamente path per label.txt
    label_txt_path = os.path.splitext(args.output_path)[0] + "_labels.txt"

    main(args.vertebrae_folder, args.disc_path, args.output_path, label_txt_path)
