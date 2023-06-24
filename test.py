import numpy as np
import pandas as pd
import dask.array as da


def FeatureExtractor(X, size, shift, axis, signal=1):
    data = None
    if axis is None:
        data = X.flatten()
    else:
        for i in reversed(range( size)):
            row = np.roll(X, shift = signal * (i + shift), axis = axis).flatten()
            
            if data is None or data.size == 0:                  #First iteration
                data = row
            else:
                data = np.vstack((data, row))
            
        for i in range(size):
            row = np.roll(X, shift = signal * -(i+shift), axis = axis).flatten()            
            data = np.vstack((data, row))
    return data
# Example usage

matrix = np.array([[[13, 13, 13, 13, 13], [13, 13, 13, 13, 13], [13, 13, 13, 13, 13]],
                    [[12, 12, 12, 12, 12], [12, 12, 12, 12, 12], [12, 12, 12, 12, 12]],
                    [[11, 11, 11, 11, 11], [11, 11, 11, 11, 11], [11, 11, 11, 11, 11]],
                    [[4, 5, 2, 6, 7], [4, 5, 1, 6, 7], [4, 5, 3, 6, 7]],
                    [[10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10]],
                    [[9, 9, 9, 9, 9], [9, 9, 9, 9, 9], [9, 9, 9, 9, 9]],
                    [[8, 8, 8, 8, 8], [8, 8, 8, 8, 8], [8, 8, 8, 8, 8]]])

x = 1
y = 1
z = 1

a = np.array([])

a = [FeatureExtractor(matrix, size=1, shift=0, axis=None)]
a = np.append(a, FeatureExtractor(matrix, size=1, shift=1, axis=1), axis=0)
a = np.append(a, FeatureExtractor(matrix, size=2, shift=1, axis=2), axis=0)
a = np.append(a, FeatureExtractor(matrix, size=3, shift=1, axis=0, signal = -1), axis=0)

print(pd.DataFrame(a))

