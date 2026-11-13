
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# File paths
ct_path = "data/spleen_2 (1).nii"

mask_path = (
    "data/segmentations/"
    "predicted_spleen_mask_52795c13-5dc7-42cb-8fc8-6452d52dd7aa.nii.gz"
)

# Load original CT and predicted mask
ct_image = nib.load(ct_path)
mask_image = nib.load(mask_path)

ct = ct_image.get_fdata()
mask = mask_image.get_fdata()

# Print information
print("Original CT shape:", ct.shape)
print("Predicted mask shape:", mask.shape)
print("Foreground voxels:", np.count_nonzero(mask))

# Verify dimensions
if ct.shape != mask.shape:
    raise ValueError("CT and predicted mask dimensions do not match!")

print("CT and mask dimensions match successfully!")

# Find slice containing the largest predicted region
slice_areas = np.count_nonzero(mask, axis=(0, 1))
slice_index = int(np.argmax(slice_areas))

print("Selected slice:", slice_index)

# Display original CT and segmentation overlay
plt.figure(figsize=(14, 6))

# Original CT
plt.subplot(1, 2, 1)
plt.imshow(
    ct[:, :, slice_index].T,
    cmap="gray",
    origin="lower"
)
plt.title(f"Original CT - Slice {slice_index}")
plt.axis("off")

# Predicted segmentation
plt.subplot(1, 2, 2)

plt.imshow(
    ct[:, :, slice_index].T,
    cmap="gray",
    origin="lower"
)

overlay = np.ma.masked_where(
    mask[:, :, slice_index].T == 0,
    mask[:, :, slice_index].T
)

plt.imshow(
    overlay,
    cmap="Reds",
    alpha=0.45,
    origin="lower"
)

plt.title("Predicted Spleen Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
