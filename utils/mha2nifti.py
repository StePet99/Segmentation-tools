import os
from glob import glob
import SimpleITK as itk

# Function to split a file path into directory, base name, and extension
def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

# Convert .mha files to .nii.gz format
def convert_folder_mha_to_nifti(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mha_files = glob(os.path.join(input_dir, '*.mha'))
    if not mha_files:
        raise FileNotFoundError(f"No .mha files found in: {input_dir}")

    for fn in mha_files:
        print(f'Converting image: {fn}')
        img = itk.ReadImage(fn)
        _, base, _ = split_filename(fn)
        out_fn = os.path.join(output_dir, base + '.nii.gz')
        itk.WriteImage(img, out_fn)
        print(f'Saved to: {out_fn}')
