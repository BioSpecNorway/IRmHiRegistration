import random

import numpy as np
import h5py
import matplotlib.pyplot as plt


class Spectra:
    wavenumbers = None
    data = None

    def __init__(self, data, wavenumbers):
        self.data = data
        self.wavenumbers = wavenumbers

    @staticmethod
    def read(filename):
        f = h5py.File(filename, "r")

        data = f["C"][:]
        wavenumbers = f["WN"][:]

        return Spectra(data, wavenumbers)

    @staticmethod
    def read_shape(filename):
        """Returns the shape of spectra array by reading it from h5"""

        f = h5py.File(filename, "r")

        shape = f["C"].shape
        return shape

    @staticmethod
    def read_wavenumbers_shape(filename):
        """Returns the shape of wavenumbers by reading it from h5"""

        f = h5py.File(filename, "r")

        return f["WN"].shape

    def write(self, filename):
        f = h5py.File(filename, "w")

        f.create_dataset("C", data=self.data, dtype=np.float16, compression="gzip")
        f.create_dataset(
            "WN", data=self.wavenumbers, dtype=np.float16, compression="gzip"
        )

        f.close()

    def show(self, n=1, labels=None):
        self.__draw(n, labels)
        plt.show()

    def __draw(self, n=1, labels=None):
        plt.figure(1, (18, 14))
        ax = plt.gca()
        ax.invert_xaxis()

        print("Spectra count: ", self.data.shape[0] if n == -1 else n)
        if n == -1:
            for i in range(self.data.shape[0]):
                if not labels:
                    plt.plot(self.wavenumbers, self.data[i].astype('float'))
                else:
                    plt.plot(self.wavenumbers, self.data[i].astype('float'), label=labels[i])
        else:
            for i in random.sample(range(self.data.shape[0]), n):
                plt.plot(self.wavenumbers, self.data[i])

        plt.ylabel("Absorption")
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        plt.legend()

    def savefig(self, filename, n=1, labels=None):
        self.__draw(n, labels)
        plt.savefig(filename)
        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()


class SpectraStatistics:
    def __init__(self, mean, std, wavenumbers, image_name=""):
        self.mean = mean
        self.std = std
        self.wavenumbers = wavenumbers
        self.image_name = " " + image_name if image_name else image_name

    @staticmethod
    def read(filename, image_name=""):
        f = h5py.File(filename, "r")

        mean = f["MEAN"][:]
        std = f["STD"][:]
        wavenumbers = f["WN"][:]

        return SpectraStatistics(mean, std, wavenumbers, image_name)

    def write(self, filename):
        f = h5py.File(filename, "w")

        f.create_dataset("MEAN", data=self.mean, dtype=np.float16, compression="gzip")
        f.create_dataset("STD", data=self.std, dtype=np.float16, compression="gzip")
        f.create_dataset(
            "WN", data=self.wavenumbers, dtype=np.float16, compression="gzip"
        )

        f.close()

    def __draw(self, sigms=1):
        plt.figure(1, (18, 14))
        ax = plt.gca()
        ax.invert_xaxis()

        plt.plot(self.wavenumbers, self.mean, label="Mean spectra")
        if sigms == 3:
            plt.fill_between(
                self.wavenumbers,
                self.mean - 3 * self.std,
                self.mean + 3 * self.std,
                label="3 sigma region",
                interpolate=True,
                color="#DCDCDC",
            )
        if sigms == 2:
            plt.fill_between(
                self.wavenumbers,
                self.mean - 2 * self.std,
                self.mean + 2 * self.std,
                label="2 sigma region",
                interpolate=True,
                color="#D3D3D3",
            )
        if sigms == 1:
            plt.fill_between(
                self.wavenumbers,
                self.mean - self.std,
                self.mean + self.std,
                label="1 sigma region",
                interpolate=True,
                color="#C0C0C0",
            )

        plt.ylabel("Absorption")
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        plt.legend()
        plt.title(f"Spectra distribution in image{self.image_name}")

    def show(self, sigms=1):
        self.__draw(sigms)
        plt.show()

    def savefig(self, filename, sigms=1):
        self.__draw(sigms)
        plt.savefig(filename)
        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()
