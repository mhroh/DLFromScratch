import numpy as np

def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def cross_entropy_error_common(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.shape[0]:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return y, t, batch_size

""" numpy matrix concept (row, column, axis)
http://taewan.kim/post/numpy_sum_axis/#fnref:1
"""
def cross_entropy_error_for_batch(y, t):
    new_y, new_t, batch_size = cross_entropy_error_common(y, t)
    return -np.sum(new_t * np.log(new_y)) / batch_size

def cross_entropy_error_for_batch_not_hotencoding(y, t):
    new_y, new_t, batch_size = cross_entropy_error_common(y, t)
    return -np.sum(np.log(new_y[np.arange(batch_size), new_t])) / batch_size

