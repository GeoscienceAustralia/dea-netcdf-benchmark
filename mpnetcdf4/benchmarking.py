import numpy as np
import h5py
import rasterio

from .utils import (
    block_iterator,
    shape_from_slice,
    select_all,
    norm_selection,
    dst_from_src,
)


def local_read(fname, varname, src_roi=None, buffer=None, offset=0):
    with h5py.File(fname, 'r') as f:
        ds = f[varname]
        src_shape = ds.shape

        if src_roi is None:
            src_roi = select_all(src_shape)

        dst_shape = shape_from_slice(src_roi, src_shape)
        dd = np.ndarray(dst_shape, dtype=ds.dtype, buffer=buffer, offset=offset)
        ds.read_direct(dd, src_roi)
        return dd


def local_block_read(fname, varname, src_roi=None, buffer=None, offset=0):
    with h5py.File(fname, 'r') as f:
        ds = f[varname]
        src_shape = ds.shape

        if src_roi is None:
            src_roi = select_all(src_shape)

        dst_shape = shape_from_slice(src_roi, src_shape)
        dst_roi = dst_from_src(src_roi, src_shape)

        dd = np.ndarray(dst_shape, dtype=ds.dtype, buffer=buffer, offset=offset)

        for roi in block_iterator(ds.chunks, src_roi, src_shape):
            ds.read_direct(dd, roi, dst_roi(roi))

        return dd


def local_read_rio(fname, varname, src_roi=None):
    assert (src_roi is None) or (len(src_roi) == 3)

    def slice_to_window(sel):
        assert sel.step in [1, None]
        return (sel.start, sel.stop)

    def slice_to_index(sel):
        #  bands use 1 based indexing
        return tuple(range(sel.start+1, sel.stop+1, sel.step))

    fname = 'NETCDF:' + fname + ':' + varname

    with rasterio.open(fname, 'r') as f:
        if src_roi is None:
            return f.read()

        src_shape = (f.count, ) + f.shape
        src_roi = norm_selection(src_roi, src_shape)

        t = slice_to_index(src_roi[0])
        window = tuple(map(slice_to_window, src_roi[1:3]))

        return f.read(indexes=t, window=window)
