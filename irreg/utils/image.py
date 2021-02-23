import numpy as np
import h5py
import imagesize

from PIL import Image as ImagePil

ImagePil.MAX_IMAGE_PIXELS = None

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from irreg.normalizations import min_max_scaler

cmap_eosin = LinearSegmentedColormap.from_list("mycmap", ["white", "darkviolet"])
cmap_hema = LinearSegmentedColormap.from_list("mycmap", ["white", "navy"])

class Image:
    data = None

    def __init__(self, data):
        self.data = data

    @staticmethod
    def read(filename):
        img = ImagePil.open(filename)
        img = np.asarray(img)
        return Image(img)

    @staticmethod
    def read_dimensions(filename):
        """Returns width and height of image by NOT reading the whole image"""
        return imagesize.get(filename)

    def write(self, filename):
        ImagePil.fromarray(min_max_scaler(self.data, 0, 255).astype("uint8")).save(
            filename
        )

    def show(self):
        plt.imshow(self.data)
        plt.show()


class HematoxylinEosinDeconvolutionVisualizer:
    @staticmethod
    def show(he_image, hematoxylin_image, eosin_image):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(he_image)
        ax[0].set_title("Original image")

        ax[1].imshow(hematoxylin_image, cmap=cmap_hema)
        ax[1].set_title("Hematoxylin")

        ax[2].imshow(eosin_image, cmap=cmap_eosin)
        ax[2].set_title("Eosin")

        for a in ax.ravel():
            a.axis("off")

        fig.tight_layout()


class Spimage:
    data = None
    wavenumbers = None

    def __init__(self, data, wavenumbers):
        self.wavenumbers = wavenumbers
        self.data = data

    @staticmethod
    def read(filename):
        f = h5py.File(filename, "r")

        data = f["C"][:]
        wavenumbers = f["WN"][:]

        return Spimage(data, wavenumbers)

    @staticmethod
    def read_dimensions(filename):
        """Returns width and height of image by reading the shape from h5"""

        f = h5py.File(filename, "r")

        shape = f["C"].shape
        return shape[:2][::-1]  # reversed shape

    @staticmethod
    def read_wavenumbers_dimensions(filename):
        """Returns the number of wavenumbers by reading it from h5"""

        f = h5py.File(filename, "r")

        return len(f["WN"])

    @staticmethod
    def read_wavenumbers(filename):
        f = h5py.File(filename, "r")

        return f["WN"][:]

    def write(self, filename):
        f = h5py.File(filename, "w")

        f.create_dataset("C", data=self.data, dtype=np.float16, compression="gzip")
        f.create_dataset(
            "WN", data=self.wavenumbers, dtype=np.float16, compression="gzip"
        )

        f.close()

    def show(self, ind=0):
        plt.imshow(self.data[:, :, ind].astype("float"))
        plt.show()
