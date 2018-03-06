from types import SimpleNamespace
import numpy as np
import h5py
import rasterio
import xarray as xr

from .utils import (
    Timer,
    block_iterator,
    shape_from_slice,
    select_all,
    norm_selection,
    dst_from_src,
)


def size_in_bytes(o):
    if isinstance(o, (np.ndarray, xr.DataArray)):
        return o.size*o.dtype.itemsize
    if isinstance(o, dict):
        return size_in_bytes(o.values())
    if isinstance(o, xr.Dataset):
        return size_in_bytes([da.values for da in o.data_vars.values()])

    return sum(size_in_bytes(x) for x in o)


def with_stats(f, message=None):
    MB = (1 << 20)
    GB = (1 << 30)

    def run(*args, **kwargs):
        with Timer() as t:
            out = f(*args, **kwargs)

        nbytes = size_in_bytes(out)
        stats = SimpleNamespace(elapsed=t.elapsed,
                                ms=t.elapsed*1000,
                                nbytes=nbytes,
                                mb=(nbytes/MB),
                                gb=(nbytes/GB),
                                throughput=nbytes/t.elapsed,
                                throughput_mb=(nbytes/t.elapsed)/MB)

        if message is not None:
            print('%s: read %.3fMb in %.5f secs, %.5f Mb/s' % (
                message,
                stats.mb,
                stats.elapsed,
                stats.throughput_mb))
        return out, stats

    return run


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


def plot_benchmark_results(stats, fig,
                           max_throughput=None,
                           max_time=None):
    st = np.r_[[(s.params.nprocs, s.t_total, s.elapsed, s.mb) for s in stats]]

    nprocs, t_total, t_read, mb = st.T
    base_throughput = mb[0]/t_total[0]

    x_ax_max = nprocs.max() + 0.5

    def set_0_based(ax, max_val=None):
        if max_val is None:
            max_val = ax.axis()[-1]
        ax.axis([0.5, x_ax_max, 0, max_val])

    axs = fig.subplots(1, 3)

    ax = axs[0]
    ax.plot(nprocs, mb/t_total, '.-')

    ax.set_title('Throughput')
    ax.yaxis.set_label_text('Mb/s')
    ax.xaxis.set_label_text('# Worker Threads')
    set_0_based(ax, max_throughput)

    ax = axs[1]
    ax.set_title('Time to Read')
    ax.plot(nprocs, t_total, '.-')
    ax.yaxis.set_label_text('secs')
    ax.xaxis.set_label_text('# Worker Threads')
    set_0_based(ax, max_time)
    #ax.axis([0.5, x_ax_max, 0, 175])

    ax = axs[2]
    ax.set_title('Efficiency per worker')
    ax.plot(nprocs, 100*(mb/t_total)/nprocs/base_throughput, '.-')
    ax.yaxis.set_label_text('%')
    ax.xaxis.set_label_text('# Worker Threads')
    set_0_based(ax, 110)
