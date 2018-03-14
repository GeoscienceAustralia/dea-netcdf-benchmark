import warnings
import numpy as np
import xarray as xr
import logging
from multiprocessing.sharedctypes import RawArray
import concurrent.futures
import threading
from collections import namedtuple, OrderedDict
from types import SimpleNamespace

from .utils import (NamedObjectCache,
                    shape_from_slice,
                    array_memsize,
                    SharedBufferView,
                    ChunkStoreAlloc,
                    select_all,
                    block_iterator,
                    dst_from_src)

warnings.filterwarnings('ignore', module="h5py")
_LOG = logging.getLogger(__name__)

NetcdfFileInfo = namedtuple('NetcdfFileInfo', ['bands', 'dims', 'grids'])
DimensionInfo = namedtuple('DimensionInfo', ['name', 'shape', 'dtype'])
BandInfo = namedtuple('BandInfo', ['name', 'shape', 'dtype', 'dims', 'chunks', 'grid_mapping', 'nodata', 'units'])


def _parse_info(info_dict):
    """ Converts dictionaries into named tuples
    """
    def mk_band_info(name, shape=None, dtype=None, dims=None, chunks=None, grid_mapping=None, nodata=None, units=None):
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

            def maybe_add(**kwargs):
                for k, v in kwargs.items():
                    if v is not None:
                        props[k] = v

            maybe_add(grid_mapping=h5_extract_attrs(ds, 'grid_mapping'),
                      nodata=h5_extract_attrs(ds, '_FillValue', unwrap_arrays=True),
                      units=h5_extract_attrs(ds, 'units', unwrap_arrays=True))

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
    # pylint: disable=protected-access
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
    def __init__(self, fname, proc, info=None, fd=None):
        self._fd = -1
        self._proc = proc

        if fd is None:
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

    @staticmethod
    def parallel_open(fname, procs):
        fds_futures = [proc.submit(eh5_open, fname) for proc in procs]
        concurrent.futures.wait(fds_futures)
        fds = [f.result() for f in fds_futures]

        # Create first one separately to get info structure
        f = NetcdfProcProxy(fname, procs[0], fd=fds[0])

        return [f] + [NetcdfProcProxy(fname, proc, info=f.info, fd=fd)
                      for proc, fd in zip(procs[1:], fds[1:])]

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


class RoundRobinSelector(object):
    def __init__(self, procs):
        self._procs = [SimpleNamespace(proc=proc, n=0) for proc in procs]

    def _pick(self):
        if len(self._procs) == 1:
            return self._procs[0]

        return min(*self._procs, key=lambda p: p.n)

    def _done_callback(self, proc):
        def on_done(_):
            proc.n -= 1
        return on_done

    def current_load(self):
        """ Takes snapshot of the current state (it's changing all the time)
        Returns a tuple of
        - Total tasks in flight
        - Number of tasks for the least busy proc
        """
        nn = [p.n for p in self._procs]
        return sum(nn), min(nn)

    def __call__(self, *args, **kwargs):
        proc = self._pick()
        future = proc.proc(*args, **kwargs)
        proc.n += 1
        future.add_done_callback(self._done_callback(proc))
        return future


class AsyncDataSink(object):
    def __init__(self):
        self._scheduled = set()

    @staticmethod
    def pack_user_data(future, slot, my_view, dst, roi):
        # pylint: disable=protected-access
        future._userdata = (slot, my_view, dst, roi)
        return future

    @staticmethod
    def unpack_user_data(future):
        # pylint: disable=protected-access
        slot, my_view, dst, roi = future._userdata
        return (slot, my_view, dst, roi)

    @staticmethod
    def _finalise(future):
        ok = False
        slot, my_view, dst, roi = AsyncDataSink.unpack_user_data(future)
        try:
            if future.result():
                dst[roi] = my_view
                ok = True
            else:
                _LOG.error('Failed one of the reads: %s %d', repr(roi), slot.id)
        except Exception as e:  # pylint: disable=broad-except
            _LOG.error('Failed with exception one of the reads: %s %d', repr(roi), slot.id)
        finally:
            slot.release()

        return ok

    def process_results(self, timeout=None, all_completed=False):
        return_when = 'ALL_COMPLETED' if all_completed else 'FIRST_COMPLETED'
        xx = concurrent.futures.wait(self._scheduled, return_when=return_when, timeout=timeout)
        for r in xx.done:
            self._finalise(r)
        self._scheduled = xx.not_done

    def drain_results_queue(self):
        """Rather than waiting for all to complete followed by large final copy of in
        flight data into the destination, interleave waiting and completing
        tasks for better latency.
        """
        while len(self._scheduled) > 0:
            self.process_results()

    def add(self, future):
        """ Append "packed" future to the queue
        """
        self._scheduled.add(future)


class MultiProcNetcdfReader(object):
    @staticmethod
    def _prepare(fname, procs):
        ff = NetcdfProcProxy.parallel_open(fname, procs)
        return ff, ff[0].info

    def __init__(self, fname, procs, state):
        """ Don't use directly, see `ReaderFactory.open`
        """
        self._fname = fname
        self._state = state
        (self._ff,
         self._info) = self._prepare(fname, procs)
        self._coords = {}

    @property
    def info(self):
        return self._info

    def load_coords(self, names=None):
        """ Load coordinates into internal cache

        :param [str]|str names: Names of dimensions to load coordinates for
        """
        if names is None:
            names = list(self._info.dims)
        if isinstance(names, str):
            names = [names]

        offset = 0
        f = self._ff[0]

        def read(dim):
            my_view = self._state.view.asarray(offset, dim.shape, dim.dtype)

            if f.read_to_shared(dim.name, np.s_[:], offset=offset).result():
                return my_view.copy()
            else:
                raise IOError('Failed to read dimension: %s' % dim.name)

        for n in names:
            assert n in self._info.dims, "No dimension named: " + n

            if n not in self._coords:
                self._coords[n] = read(self._info.dims[n])

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

    def _get_coords_values(self, dim, roi):
        dim_shape = shape_from_slice(roi, dim.shape)

        if dim.name not in self._coords and dim_shape == dim.shape:
            self.load_coords(dim.name)

        if dim.name in self._coords:
            # Use cache
            return self._coords[dim.name][roi]

        # Load just the needed data
        f = self._ff[0]
        offset = 0
        my_view = self._state.view.asarray(offset, dim_shape, dim.dtype)

        if not f.read_to_shared(dim.name, roi, offset=offset).result():
            raise IOError('Failed to read dimension: %s' % dim.name)

        return my_view.copy()

    def read_coords(self, measurement, src_roi=None):
        """ Read or lookup from cache coordinates for a given measurement or dimension

        :param str measurement: Name of measurement or dimension for which to read coordinates

        :param src_roi: Optional region of interest. Output of `np.s_[...]`.

        :returns OrderedDict(str->numpy.ndarray): Ordered dictionary mapping dimension name to coordinate values.
        """
        assert measurement in self._info.bands or measurement in self._info.dims, "No such measurement/dimension"
        if measurement in self._info.dims:
            dims = [self._info.dims[measurement]]
        else:
            dims = [self._info.dims[d] for d in self._info.bands[measurement].dims]

        if isinstance(src_roi, slice):
            src_roi = (src_roi,)

        out = OrderedDict()

        for i, dim in enumerate(dims):
            roi = slice(None, None) if src_roi is None else src_roi[i]

            out[dim.name] = self._get_coords_values(dim, roi)
        return out

    def _empty_coords(self, measurement, src_roi=None):
        assert measurement in self._info.bands or measurement in self._info.dims, "No such measurement/dimension"
        if measurement in self._info.dims:
            dims = [self._info.dims[measurement]]
        else:
            dims = [self._info.dims[d] for d in self._info.bands[measurement].dims]

        if isinstance(src_roi, slice):
            src_roi = (src_roi,)

        out = OrderedDict()

        for i, dim in enumerate(dims):
            roi = slice(None, None) if src_roi is None else src_roi[i]
            shape = shape_from_slice(roi, dim.shape)
            out[dim.name] = np.zeros(shape, dtype=dim.dtype)

        return out

    def update_coords(self, ds, src_roi=None):
        """Given a pre-allocated dataset of compatible shape update just coordinate
        values for a given ROI.

        :param xarray.Dataset ds: Pre-allocated dataset
        :src_roi: Region of interest (in source file pixel coordinates)
        """
        if src_roi is None:
            src_roi = tuple(slice(None, None) for _ in ds.coords)
        elif isinstance(src_roi, slice):
            src_roi = (src_roi,)

        for dim_name, roi in zip(ds.coords, src_roi):
            assert dim_name in self._info.dims, "No such dimension {}".format(dim_name)
            ds.coords[dim_name] = self._get_coords_values(self._info.dims[dim_name], roi)

        return ds

    def _attrs(self, m):
        attrs = {}
        if m.nodata is not None:
            attrs['nodata'] = m.nodata

        if m.units is not None:
            attrs['units'] = m.units

        if m.grid_mapping is not None:
            assert m.grid_mapping in self._info.grids, "Mismatched grid mapping reference: " + m.grid_mapping
            grid = self._info.grids[m.grid_mapping]
            crs_wkt = grid.get('crs_wkt')
            if crs_wkt is not None:
                attrs['crs'] = crs_wkt

        return attrs

    def _allocate(self, measurements, shape, dims):
        def data_array(m):
            return xr.DataArray(np.ndarray(shape, dtype=m.dtype),
                                name=m.name,
                                dims=list(dims),
                                coords=dims,
                                attrs=self._attrs(m))

        return xr.Dataset({m.name: data_array(m) for m in measurements})

    def allocate(self, measurements=None, src_roi=None, fill_coords=False):
        """Create `xarray.Dataset`, but not load pixel data yet. Note that it might read
        coordinate data though. Unless it was cached already.

        :param [str] measurements: List of measurements
        :param src_roi: Optionally region of interest can be supplied
        :param bool fill_coords: When True fill coordinate values, else leave them at 0

        :returns xarray.Dataset:
        :throws IOError: when reading coordinate data fails.
        """
        measurements, src_shape = self._check_measurements(measurements)

        if src_roi is None:
            src_roi = select_all(src_shape)

        dst_shape = shape_from_slice(src_roi, src_shape)

        if fill_coords:
            dims = self.read_coords(measurements[0].name, src_roi)
        else:
            dims = self._empty_coords(measurements[0].name, src_roi)

        return self._allocate(measurements, dst_shape, dims)

    def read(self,
             measurements=None,
             src_roi=None,
             dst=None,
             update_coords=True,
             chunk_scale=None):
        """Read data from a file in parallel

        :param [str] measurements: A list of measurements to read

        :param src_roi: Region of interest to read (e.g. `numpy.s_[:5,:,:]`)

        :param [xarray.Dataset] dst: Pre-allocated dataset, see `allocate`

        :param bool update_coords: Only meaningful when `dst` is supplied, by
        default will update coordinates with values for a given ROI, but can be
        skipped if coordinates already contain valid data for example.

        :param chunk_scale: A tuple of integers, a scaling factors for each
        dimension of measurements being read (e.g. (1,2,2) will read 4 block at
        a time in 2x2 spatial arrangement)

        """
        # pylint: disable=too-many-locals

        if measurements is None and (dst is not None):
            measurements = list(dst.data_vars)

        measurements, src_shape = self._check_measurements(measurements)

        if src_roi is None:
            src_roi = select_all(src_shape)

        dst_shape = shape_from_slice(src_roi, src_shape)
        dst_roi = dst_from_src(src_roi, src_shape)

        if dst is None:
            dst = self._allocate(measurements,
                                 dst_shape,
                                 self.read_coords(measurements[0].name, src_roi))
        else:
            for m in measurements:
                assert m.name in dst.data_vars, 'No slot for "{}"'.format(m.name)

                da = dst.data_vars[m.name]

                assert da.shape == dst_shape, 'Shape mismatch "{}", have {}, expect {}'.format(
                    m.name, da.shape, dst_shape)
                assert da.dtype == m.dtype, 'Data type mismatch "{}", have {}, expect {}'.format(
                    m.name, da.dtype, m.dtype)

            if update_coords:
                self.update_coords(dst, src_roi)

        def data_pump(read_to_shared, slot_alloc, read_chunk):
            pack_user_data = AsyncDataSink.pack_user_data

            def schedule_work(name, roi, slot, my_view, dst_array):
                future = read_to_shared(name, roi, slot.offset)
                return pack_user_data(future, slot, my_view, dst_array, dst_roi(roi))

            def alloc_one(shape, dtype):
                slot, my_view = slot_alloc(shape, dtype)
                while slot is None:
                    yield None
                    slot, my_view = slot_alloc(shape, dtype)
                yield slot, my_view

            def mk_worker(m, roi):
                return (lambda x: None if x is None else
                        schedule_work(m.name, roi, *x, dst[m.name].values))

            for roi in block_iterator(read_chunk, src_roi, src_shape):
                block_shape = shape_from_slice(roi)
                for m in measurements:
                    yield from map(mk_worker(m, roi),
                                   alloc_one(block_shape, m.dtype))

        read_to_shared = RoundRobinSelector([f.read_to_shared for f in self._ff])
        slot_alloc, read_chunk = self._init_alloc(measurements, chunk_scale)

        sink = AsyncDataSink()

        for future in data_pump(read_to_shared, slot_alloc, read_chunk):
            if future is not None:
                sink.add(future)
            else:
                sink.process_results(timeout=0.05)

            n_total, n_min = read_to_shared.current_load()
            if n_min > 3:
                sink.process_results(timeout=0.05)

        sink.drain_results_queue()
        return dst

    def close(self):
        """ Close the file.
        """
        for f in self._ff:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ReaderFactory(object):
    """This class maintains a pool of worker processes and shared to memory to
    communicate with them. There is a non-trivial costs in launching a new
    process, so it's best to re-use the same processing pool when working with
    many files one at a time.

    One can configure number of workers and the size of shared memory in
    megabytes. How much shared memory one needs depends on a number of factors:

    - Number of workers, more workers more memory is needed to keep work load steady
    - Size of read tasks in bytes

    Reading is performed in multiples of storage chunks aligned to the storage
    chunking regime.

    """

    def __init__(self, num_workers, mb=None):
        """Create a new instance. One can create several. It is safe to use different
        `ReaderFactory` instances from different threads concurrently.

        :param int num_workers: Number of worker threads to use

        :param int|None mb: Amount of memory in megabytes to reserve for
        communications with worker processes.

        """
        self._state = SharedState(mb=mb)
        self._procs = self._state.make_procs(num_workers)

    def open(self, fname, num_workers=None):
        """Open file for reading.

        :param str fname: path to a netcdf file

        :param int num_workers: Number of worker threads to use, should be
        smaller or equal to the number of workers configured during
        construction. This is useful when benchmarking as it allows to ignore
        "warmup" costs per process. When not supplied will use as many worker
        threads as were configured during construction.
        """
        if num_workers is None:
            return MultiProcNetcdfReader(fname, self._procs, self._state)

        assert num_workers <= len(self._procs), "Can't request that many workers"
        return MultiProcNetcdfReader(fname, self._procs[:num_workers], self._state)


def nc_open(fname, num_workers, mb=None):
    """Convenience method that creates new non-reusable worker pool and opens the
    file for parallel reading. If you need to process many files consider using
    `ReaderFactory` to amortize the cost of launching processing workers.

        :param str fname: path to a netcdf file
        :param int num_workers: Number of worker threads to use
        :param int|None mb: Amount of memory in megabytes to reserve for
        communications with worker processes.

    """
    return ReaderFactory(num_workers, mb=mb).open(fname)
