from PIL import Image
import numpy as np

def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        try:
            img.seek(i)
            slice_ = np.zeros((img.width, img.height))
            for j in range(slice_.shape[0]):
                for k in range(slice_.shape[1]):
                    slice_[j,k] = img.getpixel((j, k))
            images.append(slice_)

        except EOFError:
            # Not enough frames in img
            break

    return np.array(images), img.n_frames