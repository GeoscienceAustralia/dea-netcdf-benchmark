import warnings
import numpy as np
import logging
from multiprocessing.sharedctypes import RawArray
import concurrent.futures

from utils import (NamedObjectCache,
                   shape_from_slice,
                   array_memsize,
                   SharedBufferView,
                   ChunkStoreAlloc)

warnings.filterwarnings('ignore', module="h5py")
_LOG = logging.getLogger(__name__)


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
        keys = ds.attrs.keys()

    return {k: safe_extract(k) for k in keys}


class ExternalNetcdfReader(object):
    DEFAULT_SHARED_MEM_SZ = 40*(1 << 20)
    _shared_view = None

    @staticmethod
    def initialised():
        return ExternalNetcdfReader._shared_view is not None

    @staticmethod
    def static_init(shared_mem_size=DEFAULT_SHARED_MEM_SZ):
        ExternalNetcdfReader._shared_view = SharedBufferView(RawArray('b', shared_mem_size))

    @staticmethod
    def mk_proc():
        if ExternalNetcdfReader.initialised() is False:
            ExternalNetcdfReader.static_init()
        return concurrent.futures.ProcessPoolExecutor(max_workers=1)

    @staticmethod
    def slot_allocator(chunk_shape, dtype, cache=None):
        # TODO: only one allocator is assumed to exists/be used at a time,
        # need to enforce it

        if cache is None:
            chunk_sz = array_memsize(chunk_shape, dtype)

            if ExternalNetcdfReader._shared_view is None:
                ExternalNetcdfReader.static_init()

            cache = ChunkStoreAlloc(chunk_sz,
                                    ExternalNetcdfReader._shared_view.buffer)

        if cache.check_size(chunk_shape, dtype) is False:
            _LOG.error('Requested chunk size is too large')
            return None, cache

        def alloc(shape=chunk_shape):
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

    def __init__(self, fname):
        import h5py

        self._view = ExternalNetcdfReader._shared_view
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
            return b'DIMENSION_SCALE' == ds.attrs.get('CLASS')

        def is_nc_grid(ds):
            return 'grid_mapping_name' in ds.attrs

        def is_nc_var(ds):
            return ds.attrs.get('DIMENSION_LIST') is not None

        def nc_dims(ds):
            return [ds.file[ref[0]].name.split('/')[-1] for ref in ds.attrs['DIMENSION_LIST']]

        def describe_var(ds):
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

        vars = {k: describe_var(self._f[k])
                for k in self._f.keys()
                if is_nc_var(self._f[k])}

        dims = {k: describe_dim(self._f[k])
                for k in self._f.keys()
                if is_nc_dim(self._f[k])}

        grids = {k: describe_grid(self._f[k])
                 for k in self._f.keys()
                 if is_nc_grid(self._f[k])}

        return dict(vars=vars,
                    dims=dims,
                    grids=grids)


def eh5_open(fname):
    try:
        (fd, _) = NamedObjectCache(ExternalNetcdfReader)(fname)
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
    def __init__(self, fname, proc=None, info=None):
        self._fd = -1
        self._proc = None

        if proc is None:
            proc = ExternalNetcdfReader.mk_proc()

        fd = proc.submit(eh5_open, fname)
        fd = fd.result()

        if fd < 0:
            raise IOError('Failed to open file: "%s"', fname)

        if info is None:
            info = proc.submit(eh5_info, fd)
            info = info.result()

            if info is None:
                raise IOError('Failed to query info about the file: "%s"', fname)

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


class MultiProcNetcdfReader(object):

    class Rdr(object):
        def __init__(self, fname, procs, view):
            self._fname = fname
            self._procs = procs
            self._view = view
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

        def read(self, measurements=None,
                 src_roi=None,
                 chunk_scale=None):
            bands = self._info['vars']

            if measurements is None:
                measurements = list(bands.keys())

            largest_dtype = None
            for m in measurements:
                if m not in bands:
                    raise ValueError('No such measurement')

                dtype = np.dtype(bands[m]['dtype'])
                if (largest_dtype is None or
                   dtype.itemsize > largest_dtype.itemsize):
                    largest_dtype = dtype

            read_chunk = bands[measurements[0]]['chunks']
            if chunk_scale is not None:
                read_chunk = tuple(ch*s for ch, s in
                                   zip(read_chunk, chunk_scale))

            # TODO: should use self._view as backing store
            slot_alloc, _ = ExternalNetcdfReader.slot_allocator(read_chunk, largest_dtype)

        def close(self):
            for f in self._ff:
                f.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def __init__(self, num_workers, mem_sz=None):
        if mem_sz is not None:
            if mem_sz < 10000:
                mem_sz = mem_sz * (1 << 20)  # Assume in Mb, convert to bytes

            if ExternalNetcdfReader.initialised() is False:
                ExternalNetcdfReader.static_init(mem_sz)
            elif len(ExternalNetcdfReader._shared_view.buffer) < mem_sz:
                # TODO: assumes no one is using it already
                ExternalNetcdfReader.static_init(mem_sz)

        self._shared_view = ExternalNetcdfReader._shared_view
        self._procs = [ExternalNetcdfReader.mk_proc() for _ in range(num_workers)]
        pass

    def open(self, fname):
        return MultiProcNetcdfReader.Rdr(fname,
                                         self._procs,
                                         self._shared_view)


def test_multi():
    fname = "sample.nc"

    mpr = MultiProcNetcdfReader(2)

    f = mpr.open(fname)

    print(f._ff)
    f.close()

    with mpr.open(fname) as f:
        print(f._info)
