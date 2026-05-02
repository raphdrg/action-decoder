import zipfile
import os

zip_path = os.path.join("data", "img_align_celeba.zip")
extract_to = "data"

print(f"Extracting {zip_path} to {extract_to}/ ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_to)

# Count extracted images
img_dir = os.path.join(extract_to, "img_align_celeba")
num_images = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
print(f"Done. {num_images:,} images extracted to {img_dir}/")
