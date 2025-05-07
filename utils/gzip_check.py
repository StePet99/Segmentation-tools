import gzip
import shutil
import os

def check(file_path):
    """
    Ensure that a .nii.gz file is a valid gzip. If not, recompress it properly.
    """
    def is_valid_gzip(path):
        try:
            with gzip.open(path, 'rb') as f:
                f.read(1)  # Try reading one byte
            return True
        except Exception:
            return False

    if not file_path.endswith(".nii.gz"):
        raise ValueError("File must have .nii.gz extension")

    if is_valid_gzip(file_path):
        print(f"{file_path} is a valid gzip file.")
        return file_path

    # Not a valid gzip – fix it
    print(f"{file_path} is NOT a valid gzip file. Fixing...")

    # Rename to .nii temporarily
    uncompressed_path = file_path[:-3]
    os.rename(file_path, uncompressed_path)

    # Recompress using proper gzip
    with open(uncompressed_path, 'rb') as f_in:
        with gzip.open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Optionally delete uncompressed file
    os.remove(uncompressed_path)

    print(f"Recompressed {uncompressed_path} to valid {file_path}")
    return file_path
