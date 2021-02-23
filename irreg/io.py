import numpy as np
import h5py


def read_spimage(filename):
    f = h5py.File(filename, "r")

    data = f["C"][:]
    wavenumbers = f["WN"][:].ravel()

    return data, wavenumbers


def write_spimage(filename, data, wavenumbers):
    f = h5py.File(filename, "w")

    f.create_dataset("C", data=data, dtype=np.float16, compression="gzip")
    f.create_dataset("WN", data=wavenumbers, dtype=np.float16, compression="gzip")

    f.close()
