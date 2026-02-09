import os
import glob
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans

# =============================
# Configuration
# =============================

# Output image size (H = W = img_size)
img_size = 120

# Number of augmentations per slice
num_of_aug = 1

# =============================
# Dataset Loader
# =============================

def load_dataset(src, mask, label=False, resize=(155, img_size, img_size)):
    """
    Read NIfTI (.nii) files, resize volumes,
    normalize MRI images, binarize masks,
    slice into 2D images, apply augmentation,
    and save as NumPy arrays.
    """

    # FIX: correct path joining for Windows
    files = glob.glob(os.path.join(src, mask), recursive=True)

    imgs = []
    counter = 0

    print("Processing ---", mask)
    print("Found", len(files), "files")

    if len(files) == 0:
        print("⚠ No files found. Check your dataset path and mask.")
        return

    for file in files:
        counter += 1
        if counter == 81:   # limit number of volumes (as in original code)
            break

        print(counter, ". volume:", file)

        # Read 3D MRI volume (D, H, W)
        img = io.imread(file, plugin="simpleitk")

        # Resize volume
        img = trans.resize(
            img,
            resize,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True
        )

        if label:
            # Tumor mask → binary
            img[img != 0] = 1
            img = img.astype("float32")
        else:
            # FLAIR normalization
            mean = img.mean()
            std = img.std()
            if std != 0:
                img = (img - mean) / std
            else:
                img = img - mean

        # Slice the volume (ignore empty top/bottom slices)
        for slice_idx in range(50, 130):
            img_t = img[slice_idx, :, :]

            # Shape → (1, 1, H, W)  (channels_first)
            img_t = img_t.reshape((1, 1) + img_t.shape)

            # Data augmentation
            img_g = aug(img_t, num_of_aug)

            for n in range(img_g.shape[0]):
                imgs.append(img_g[n])

    name = "y_" + str(img_size) if label else "x_" + str(img_size)
    np.save(name, np.array(imgs, dtype="float32"))

    print("Saved", len(imgs), "slices to", name + ".npy")


# =============================
# Data Augmentation
# =============================

def aug(scans, n):
    """
    Apply random augmentation to a 4D tensor:
    (N, C, H, W)
    """

    datagen = ImageDataGenerator(
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True
    )

    img_g = scans.copy()
    i = 0

    for batch in datagen.flow(scans, batch_size=1, seed=1000):
        img_g = np.vstack([img_g, batch])
        i += 1
        if i == n:
            break

    return img_g


# =============================
# Main
# =============================

def main():
    # FIX: raw string for Windows path
    src_root = r"E:\Final_Year_Project\BraTS_small\LGG"

    load_dataset(
        src_root,
        "**/*_seg.nii",
        label=True,
        resize=(155, img_size, img_size)
    )

    load_dataset(
        src_root,
        "**/*_flair.nii",
        label=False,
        resize=(155, img_size, img_size)
    )


if __name__ == "__main__":
    main()
