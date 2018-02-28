import warnings
import numpy as np
import logging
from multiprocessing.sharedctypes import RawArray
import concurrent.futures
import threading
from collections import namedtuple

from utils import (NamedObjectCache,
                   shape_from_slice,
                   array_memsize,
                   SharedBufferView,
                   ChunkStoreAlloc)

warnings.filterwarnings('ignore', module="h5py")
_LOG = logging.getLogger(__name__)

NetcdfFileInfo = namedtuple('NetcdfFileInfo', ['bands', 'dims', 'grids'])
DimensionInfo = namedtuple('DimensionInfo', ['name', 'shape', 'dtype'])
BandInfo = namedtuple('BandInfo', ['name', 'shape', 'dtype', 'dims', 'chunks', 'grid_mapping', 'nodata'])


def _parse_info(info_dict):
    """ Converts dictionaries into named tuples
    """
    def mk_band_info(name, shape=None, dtype=None, dims=None, chunks=None, grid_mapping=None, nodata=None):
        return BandInfo(**locals())

    bands = {k: mk_band_info(k, **v) for k, v in info_dict['bands'].items()}
    dims = {k: DimensionInfo(name=k, **v) for k, v in info_dict['dims'].items()}
    grids = info_dict.get('grids')

    return NetcdfFileInfo(bands=bands, dims=dims, grids=grids)


def h5_utf8_attr(f, name):
    from h5py import h5a, h5t
    name = name.encode('ascii')
    a = h5a.open(f.id, name)
    mtype = h5t.py_create(a.dtype)
    mtype.set_cset(1)
    data = np.ndarray((), dtype=a.dtype)
    a.read(data, mtype=mtype)
    return data[()].decode('utf8')


def h5_extract_attrs(ds, keys=None, unwrap_arrays=False):
    def maybe_unwrap(v):
        if isinstance(v, np.ndarray):
            if v.shape == (1,):
                return v.tolist()[0]
            else:
                return v.tolist()
            return v

    def safe_extract(k):
        if k not in ds.attrs:
            return None

        try:
            v = ds.attrs.get(k)
        except OSError:
            return h5_utf8_attr(ds, k)

        if unwrap_arrays:
            v = maybe_unwrap(v)

        return v

    if isinstance(keys, (str,)):
        return safe_extract(keys)

    if keys is None:
        keys = list(ds.attrs)

    return {k: safe_extract(k) for k in keys}


class ExternalNetcdfReader(object):
    def __init__(self, fname, view):
        import h5py

        self._view = view
        self._f = None
        try:
            self._f = h5py.File(fname, 'r')
        except IOError as e:
            _LOG.error('Failed to open: ""%s"', fname)
            raise e

    def __del__(self):
        if self._f is not None:
            self._f.close()

    def _get_dst_buf(self, offset, dtype, shape, roi):
        dst_shape = shape_from_slice(roi, shape)

        a = self._view.asarray(offset, dst_shape, dtype)
        if a is None:
            _LOG.error('Failed to create chunk, chunk too big?')
        return a

    def read_to_shared(self, varname, src_roi, offset):
        if varname not in self._f:
            _LOG.error('No dataset named "%s"', varname)
            return False

        ds = self._f[varname]

        out = self._get_dst_buf(offset, ds.dtype, ds.shape, src_roi)
        if out is None:
            _LOG.error('Failed to lookup offset: %d', offset)
            return False

        ds.read_direct(out, src_roi)
        return True

    def info(self):
        def is_nc_dim(ds):
            return ds.attrs.get('CLASS') == b'DIMENSION_SCALE'

        def is_nc_grid(ds):
            return 'grid_mapping_name' in ds.attrs

        def is_nc_var(ds):
            return ds.attrs.get('DIMENSION_LIST') is not None

        def nc_dims(ds):
            return tuple(ds.file[ref[0]].name.split('/')[-1] for ref in ds.attrs['DIMENSION_LIST'])

        def describe_band(ds):
            props = dict(shape=ds.shape,
                         chunks=ds.chunks,
                         dtype=ds.dtype.name,
                         dims=nc_dims(ds))
            grid_mapping = h5_extract_attrs(ds, 'grid_mapping')
            if grid_mapping is not None:
                props.update({'grid_mapping': grid_mapping})

            nodata = h5_extract_attrs(ds, '_FillValue', unwrap_arrays=True)
            if nodata is not None:
                props.update({'nodata': nodata})

            return props

        def describe_dim(ds):
            return dict(shape=ds.shape,
                        dtype=ds.dtype.name)

        def describe_grid(ds):
            return h5_extract_attrs(ds, unwrap_arrays=True)

        bands = {k: describe_band(self._f[k])
                 for k in self._f.keys()
                 if is_nc_var(self._f[k])}

        dims = {k: describe_dim(self._f[k])
                for k in self._f.keys()
                if is_nc_dim(self._f[k])}

        grids = {k: describe_grid(self._f[k])
                 for k in self._f.keys()
                 if is_nc_grid(self._f[k])}

        return dict(bands=bands,
                    dims=dims,
                    grids=grids)


class SharedState(object):
    DEFAULT_SHARED_MEM_SZ = 40*(1 << 20)
    _shared_view = None
    _launch_lock = threading.Lock()

    @staticmethod
    def _check_buf():
        return SharedState._shared_view is not None

    def __init__(self, view=None, mb=None):
        if view is not None:
            self._view = view
        elif mb is not None:
            self._view = SharedBufferView(RawArray('b', mb*(1 << 20)))
        else:
            self._view = SharedBufferView(RawArray('b', SharedState.DEFAULT_SHARED_MEM_SZ))

    @property
    def view(self):
        return self._view

    def make_procs(self, num_workers):
        with SharedState._launch_lock:
            # TODO: rather than putting it into this global slot give it a name and pass it through NamedObjectCache
            #       see eh5_open
            SharedState._shared_view = self._view
            pp = [concurrent.futures.ProcessPoolExecutor(max_workers=1) for _ in range(num_workers)]

            # ensure that processes are launched and global state can be released in the main process
            ff = concurrent.futures.wait([p.submit(SharedState._check_buf) for p in pp])
            results = list(r.result() for r in ff.done)
            SharedState._shared_view = None

        if set(results) != set([True]):
            raise RuntimeError('Failed to share work buffer with worker processes')

        return pp

    def slot_allocator(self, chunk_shape, dtype):
        chunk_sz = array_memsize(chunk_shape, dtype)

        cache = ChunkStoreAlloc(chunk_sz, self._view)

        if cache.check_size(chunk_shape, dtype) is False:
            _LOG.error('Requested chunk size is too large')
            return None, cache

        def alloc(shape=chunk_shape, dtype=dtype):
            empty = (None, None)

            slot = cache.alloc()
            if slot is None:
                return empty

            a = cache.asarray(slot, shape, dtype)

            if a is None:
                del slot
                _LOG.error('Something went wrong: chunk is too small suddenly')
                raise ValueError('Shape too big I guess, or unexpected state changes')

            return (slot, a)

        return alloc, cache


def eh5_open(fname, view=None):
    if view is None:
        view = SharedState._shared_view
        if view is None:
            return -2

    try:
        (fd, _) = NamedObjectCache(lambda fname: ExternalNetcdfReader(fname, view))(fname)
    except IOError as e:
        return -1

    return fd


def eh5_read_to_shared(fd, varname, src_roi, offset):
    f = NamedObjectCache().lookup(fd)
    if f is None:
        return False

    return f.read_to_shared(varname, src_roi, offset)


def eh5_info(fd):
    f = NamedObjectCache().lookup(fd)
    if f is None:
        return None
    return f.info()


def eh5_close(fd):
    f = NamedObjectCache().clear_by_id(fd)
    if f is None:
        return False
    del f
    return True


class NetcdfProcProxy(object):
    def __init__(self, fname, proc, info=None):
        self._fd = -1
        self._proc = proc

        fd = proc.submit(eh5_open, fname)
        fd = fd.result()

        if fd < 0:
            raise IOError('Failed to open file: "%s"' % fname)

        if info is None:
            info = proc.submit(eh5_info, fd)
            info = info.result()

            if info is None:
                raise IOError('Failed to query info about the file: "%s"' % fname)

            info = _parse_info(info)

        self._proc = proc
        self._fd = fd
        self._info = info

    @property
    def proc(self):
        return self._proc

    @property
    def info(self):
        return self._info

    def close(self):
        return self._proc.submit(eh5_close, self._fd).result()

    def read_to_shared(self, varname, roi, offset):
        return self._proc.submit(eh5_read_to_shared, self._fd, varname, roi, offset)


class _MultiProcNetcdfReader(object):
    def __init__(self, fname, procs, state):
        self._fname = fname
        self._procs = procs
        self._state = state
        self._ff = []
        self._info = None
        self._prepare()

    def _prepare(self):
        ff = []
        info = None
        for i, proc in enumerate(self._procs):
            f = NetcdfProcProxy(self._fname, proc=proc, info=info)
            ff.append(f)
            if i == 0:
                info = f.info

        self._ff = ff
        self._info = info

    def _check_measurements(self, measurements):
        bands = self._info.bands

        if isinstance(measurements, str):
            measurements = [measurements]

        if measurements is None:
            measurements = [k for k in bands if k != 'dataset']  # TODO: GA specifics 'dataset'
        else:
            for m in measurements:
                if m not in bands:
                    raise ValueError('No such measurement: ' + m)

        measurements = [bands[m] for m in measurements]

        # TODO: for now just checking shapes, but should check dimension names
        # and grid_mapping
        shapes = set(map(lambda m: m.shape, measurements))
        if len(shapes) != 1:
            raise ValueError('Expect all requested bands to have the same shape')

        return measurements, shapes.pop()

    def _init_alloc(self, measurements, chunk_scale):
        dtypes = map(lambda m: np.dtype(m.dtype), measurements)
        dtypes = sorted(dtypes, key=lambda dtype: dtype.itemsize, reverse=True)
        largest_dtype = dtypes[0]

        read_chunk = measurements[0].chunks
        if chunk_scale is not None:
            read_chunk = tuple(ch*s for ch, s in
                               zip(read_chunk, chunk_scale))

        slot_alloc, _ = self._state.slot_allocator(read_chunk, largest_dtype)
        return slot_alloc, read_chunk

    def read(self, measurements=None,
             src_roi=None,
             chunk_scale=None):
        from utils import select_all

        measurements, src_shape = self._check_measurements(measurements)

        if src_roi is None:
            src_roi = select_all(src_shape)

        slot_alloc, read_chunk = self._init_alloc(measurements, chunk_scale)

        return slot_alloc, read_chunk

    def close(self):
        for f in self._ff:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MultiProcNetcdfReader(object):

    def __init__(self, num_workers, mb=None):
        self._state = SharedState(mb=mb)
        self._procs = self._state.make_procs(num_workers)

    def open(self, fname):
        return _MultiProcNetcdfReader(fname, self._procs, self._state)


def test_multi():
    fname = "sample.nc"

    mpr = MultiProcNetcdfReader(2)

    f = mpr.open(fname)

    f.close()

    with mpr.open(fname) as f:
        xx = f.read()
        print(xx)
        assert xx is not None
