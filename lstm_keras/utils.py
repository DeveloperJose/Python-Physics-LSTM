import numpy as np

def shift_range(arr, new_range):
    fm = arr - np.min(arr)
    fd = fm / np.max(fm)
    return (new_range-1) * fd