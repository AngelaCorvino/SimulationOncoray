import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_dose(directory, filename, m):
    # data = pd.read_excel (path)
    """read reconstruction  data from npy document"""
    data = np.load(directory + filename)

    dose = data[0 : len(data) - 3]  # this is usually the dose
    depth = np.arange(0, len(dose), 1) * m  # m = [mm/pixe]it depends on the camera
    tof = data[len(data) - 3 : len(data)]
    (prefix, shotname) = filename.split("_")
    (shotname, extension) = shotname.split(".npy")
    (array, shotnumber) = shotname.split("array")
    return dose, depth, tof, shotnumber


def read_doserr(directory, filename):
    # data = pd.read_excel (path)
    """read reconstruction  data from npy document"""
    data = np.load(directory + filename)
    err = data[0 : len(data)]  # this is usually the dose

    return err
