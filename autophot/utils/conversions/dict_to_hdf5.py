from typing import Iterable


def to_hdf5_has_None(l):
    for i in range(len(l)):
        if hasattr(l[i], "__iter__") and not isinstance(l[i], str):
            l[i] = to_hdf5_has_None(l[i])
        elif l[i] is None:
            return True
    return False


def dict_to_hdf5(h, D):
    for key in D:
        if isinstance(D[key], dict):
            n = h.create_group(key)
            dict_to_hdf5(n, D[key])
        else:
            if hasattr(D[key], "__iter__") and not isinstance(D[key], str):
                if to_hdf5_has_None(D[key]):
                    h[key] = str(D[key])
                else:
                    h.create_dataset(key, data=D[key])
            elif D[key] is not None:
                h[key] = D[key]
            else:
                h[key] = "None"


def hdf5_to_dict(h):
    import h5py

    D = {}
    for key in h.keys():
        if isinstance(h[key], h5py.Group):
            D[key] = hdf5_to_dict(h[key])
        elif isinstance(h[key], str) and "None" in h[key]:
            D[key] = eval(h[key])
        else:
            D[key] = h[key]
    return D
