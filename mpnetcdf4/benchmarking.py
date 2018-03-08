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


def local_read(fname, bands, src_roi=None):

    def read_band(ds, src_roi):
        src_shape = ds.shape

        if src_roi is None:
            src_roi = select_all(src_shape)

        dst_shape = shape_from_slice(src_roi, src_shape)
        dd = np.ndarray(dst_shape, dtype=ds.dtype)
        ds.read_direct(dd, src_roi)
        return dd

    if isinstance(bands, str):
        bands = [bands]

    with h5py.File(fname, 'r') as f:
        return {name: read_band(f[name], src_roi)
                for name in bands}


def local_block_read(fname, bands, src_roi=None, chunk_scale=None):
    def read_band(ds, src_roi):
        src_shape = ds.shape

        if src_roi is None:
            src_roi = select_all(src_shape)

        read_chunk = ds.chunks
        if chunk_scale is not None:
            read_chunk = tuple(ch*s for ch, s in
                               zip(read_chunk, chunk_scale))

        dst_shape = shape_from_slice(src_roi, src_shape)
        dst_roi = dst_from_src(src_roi, src_shape)

        dd = np.ndarray(dst_shape, dtype=ds.dtype)

        for roi in block_iterator(read_chunk, src_roi, src_shape):
            ds.read_direct(dd, roi, dst_roi(roi))

        return dd

    if isinstance(bands, str):
        bands = [bands]

    with h5py.File(fname, 'r') as f:
        return {name: read_band(f[name], src_roi)
                for name in bands}


def local_read_rio(fname, bands, src_roi=None):
    assert (src_roi is None) or (len(src_roi) == 3)

    def slice_to_window(sel):
        assert sel.step in [1, None]
        return (sel.start, sel.stop)

    def slice_to_index(sel):
        #  bands use 1 based indexing
        return tuple(range(sel.start+1, sel.stop+1, sel.step))

    def read_band(fname, varname, src_roi):
        fname = 'NETCDF:' + fname + ':' + varname

        with rasterio.open(fname, 'r') as f:
            if src_roi is None:
                return f.read()

            src_shape = (f.count, ) + f.shape
            src_roi = norm_selection(src_roi, src_shape)

            t = slice_to_index(src_roi[0])
            window = tuple(map(slice_to_window, src_roi[1:3]))

            return f.read(indexes=t, window=window)

    if isinstance(bands, str):
        bands = [bands]

    return {name: read_band(fname, name, src_roi)
            for name in bands}


def run_benchmark(pp, mp_factory=None, dst=None, **pp_overrides):
    """
    pp.fname        -- File to read
    pp.measurements -- Which bands to load
    pp.src_roi      -- Region of interest to read, can be None
    pp.nprocs       -- Number of processes to use
    pp.mb           -- Size of shared memory in Mb
    pp.chunk_scale  -- Read chunk scaler, can be None to read one chunk at a time

    mp_factory      -- Optional pre-allocated `ReaderFactory`
    dst             -- Optional pre-allocated destination

    **pp_overrides  -- Override any of the above parameters before running

    :returns: out, stats
    """
    from copy import copy
    from .ncread import ReaderFactory
    from .utils import Timer

    pp = copy(pp)

    for k, v in pp_overrides.items():
        setattr(pp, k, v)

    with Timer() as t_total:
        if mp_factory is None:
            with Timer(message='Prepare x%d' % pp.nprocs) as t:
                mp_factory = ReaderFactory(pp.nprocs, mb=pp.mb)

            t_prepare = t.elapsed
        else:
            t_prepare = 0

        with Timer(message='Open x%d' % pp.nprocs) as t:
            f = mp_factory.open(pp.fname, pp.nprocs)

        t_open = t.elapsed

        read = with_stats(f.read, message='Read x%d' % pp.nprocs)

        out, stats = read(measurements=pp.measurements,
                          src_roi=pp.src_roi,
                          dst=dst,
                          chunk_scale=pp.chunk_scale)

    stats.t_total = t_total.elapsed
    stats.t_prepare = t_prepare
    stats.t_open = t_open
    stats.params = pp
    stats.params.dst_supplied = dst is not None

    return out, stats


def plot_benchmark_results(stats, fig,
                           max_throughput=None,
                           base_throughput=None,
                           max_time=None):
    st = np.r_[[(s.params.nprocs, s.t_total, s.elapsed, s.mb) for s in stats]]

    nprocs, t_total, t_read, mb = st.T
    if base_throughput is None:
        base_throughput = (mb/(t_total*nprocs)).max()

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

    ax = axs[2]
    ax.set_title('Efficiency per worker')
    ax.plot(nprocs, 100*(mb/t_total)/nprocs/base_throughput, '.-')
    ax.yaxis.set_label_text('%')
    ax.xaxis.set_label_text('# Worker Threads')
    set_0_based(ax, 110)


def find_next_available_file(fname_pattern, max_n=1000, start=1):
    """
    :param str fname_pattern: File name pattern using "%d" style formatting e.g. "result-%03d.png"
    :param int max_n: Check at most that many files before giving up and returning None
    :param int start: Where to start counting from, default is 1
    """
    from pathlib import Path

    for i in range(start, max_n):
        fname = fname_pattern % i
        if not Path(fname).exists():
            return fname

    return None


def dump_as_pickle(data, fname_pattern, max_n=1000, start=1):
    """
    :param data: Object to pickle
    :param str fname_pattern: File name pattern using "%d" style formatting e.g. "result-%03d.pickle"
    :param int max_n: Check at most that many files before giving up and failing
    :param int start: Where to start counting from, default is 1
    :return str: File name to which things were written
    """
    import pickle

    fname = find_next_available_file(fname_pattern, max_n=max_n, start=start)

    if fname is None:
        return None

    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    return fname
