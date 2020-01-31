import numpy as np
def normalize_t(arr, vb=False):
    norm_factor = np.sum(arr)
    arr /= norm_factor
    var = np.sum(arr)
    try:
        assert np.isclose(var, 1.)
        if vb: print(var)
    except AssertionError:
        print('normalization error in normalize_t'+str(var))
    return arr
    
def normalize_z(arr, z_difs, vb=False):
    norm_factor = np.sum(arr * z_difs)
    arr /= norm_factor
    var = np.sum(arr * z_difs)
    try:
        assert np.isclose(var, 1.)
        if vb: print(var)
    except AssertionError:
        print('normalization error in normalize_z'+str(var))
    return arr


def normalize_zt(arr, z_difs, vb=False):
    norm_factor = np.sum(arr * z_difs[np.newaxis, :])
    arr /= norm_factor
    var = np.sum(arr * z_difs[np.newaxis, :])
    try:
        assert np.isclose(var, 1.)
        if vb: print(var)
    except AssertionError:
        print('normalization error in normalize_zt'+str(var))
    return arr

def normalize_hubble(arr, z_difs, mu_difs, vb=False):
    norm_factor = np.sum(arr * z_difs[:, np.newaxis] * mu_difs[np.newaxis, :])
    arr /= norm_factor
    var = np.sum(arr * z_difs[:, np.newaxis] * mu_difs[np.newaxis, :])
    try:
        assert np.isclose(var, 1.)
        if vb: print(var)
    except AssertionError:
        print('normalization error in normalize_hubble'+str(var))
    return arr
    
def normalize_one(arr,z_difs, mu_difs, vb=False):
    norm_factor = np.sum(arr * z_difs[np.newaxis, :, np.newaxis] * mu_difs[np.newaxis, np.newaxis, :])
    arr /= norm_factor
    var = np.sum(arr * z_difs[np.newaxis, :, np.newaxis] * mu_difs[np.newaxis, np.newaxis, :])
    try:
        assert np.isclose(var, 1.)
        if vb: print(var)
    except AssertionError:
        print('normalization error in normalize_one '+str(var))
    return arr
    
def normalize_all(arr, z_difs, mu_difs, vb=False):

    nans = np.isnan(arr)
    n_objs = len(arr)
    norm_factor = np.sum(np.sum(np.sum(arr * z_difs[np.newaxis, np.newaxis, :, np.newaxis] * mu_difs[np.newaxis, np.newaxis, np.newaxis, :], axis=3), axis=2), axis=1)
    arr /= norm_factor[:, np.newaxis, np.newaxis, np.newaxis]
    var = np.sum(np.sum(np.sum(arr * z_difs[np.newaxis, np.newaxis, :, np.newaxis] * mu_difs[np.newaxis, np.newaxis, np.newaxis, :], axis=3), axis=2), axis=1)
    try:
        assert np.all(np.isclose(var, np.ones(n_objs)))
        if vb: print(var, 'hi')
    except AssertionError:
        print('normalization error in normalize_all '+str(var))
        print(arr,'arr')

    return arr
    
