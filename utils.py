import threading
import functools
import itertools
import operator
import numpy as np


class NamedObjectCache(object):
    """Maps arbitrary objects (often opened files) to numerical ids, that can later
    be looked up. This is meant to be used with multiprocessing.
    """

    _shared = type('_shared', (object,), {})()
    _shared.count = 0
    _shared.name2fd = {}
    _shared.fd2obj = {}
    _shared.lock = threading.Lock()

    def __init__(self, init=None):
        self._init = init

    def __call__(self, name, init=None):
        """ name -> (fd, init(name))
        """
        if init is None:
            init = self._init

        st = self._shared

        fd, obj = st.name2fd.get(name, (None, None))
        if fd is not None:
            return (fd, obj)

        if init is None:
            return (None, None)

        with st.lock:
            # lookup again in case other thread inited it already
            fd, obj = st.name2fd.get(name, (None, None))
            if fd is not None:
                return (fd, obj)

            obj = init(name)

            st.count += 1
            fd = st.count

            st.name2fd[name] = (fd, obj)
            st.fd2obj[fd] = (obj, name)

        return (fd, obj)

    def lookup(self, fd):
        """ fd -> previously created object or None if not found
        """
        return self._shared.fd2obj.get(fd, (None, None))[0]

    def clear_by_name(self, name):
        """ Remove named object from the cache using name as key
        """
        st = self._shared
        with st.lock:
            fd, obj = st.name2fd.pop(name, (None, None))
            if fd is None:
                return None
            obj, _ = st.fd2obj.pop(fd, (None, None))
            return obj

    def clear_by_id(self, fd):
        """ Remove named object from the cache using numerical id as a key
        """
        st = self._shared
        with st.lock:
            obj, name = st.fd2obj.pop(fd, (None, None))
            if obj is None:
                return None
            st.name2fd.pop(name, None)
            return obj


def norm_slice(s, n=None):
    if isinstance(s, (int)):
        return slice(s, s + 1, 1)

    def norm_pt(x, value_if_none):
        if x is None:
            if value_if_none is None:
                raise ValueError("Can not normalise open ended slice without knowing array shape")
            return value_if_none
        if x < 0:
            if n is None:
                raise ValueError("Can not normalise relative to end slice without knowing array shape")
            return n + x
        return x

    return slice(norm_pt(s.start, 0),
                 norm_pt(s.stop, n),
                 1 if s.step is None else s.step)


def shape_from_slice(sel, shape=None):

    def size_from_slice(s, n):
        s = norm_slice(s, n)
        if s.start >= s.stop:
            return 0

        return 1 + (s.stop - s.start - 1)//s.step

    if not isinstance(sel, (tuple,)):
        sel = (sel,)

    if shape is None:
        return tuple(size_from_slice(s, None) for s in sel)

    if not isinstance(shape, (tuple,)):
        shape = (shape,)

    return tuple(size_from_slice(s, n) for s, n in zip(sel, shape))


def offset_from_slice(sel, shape=None):
    def offset(s, n):
        if s.start is None:
            return 0
        if s.start >= 0:
            return s.start
        if n is None:
            raise ValueError("Need to know full shape to deal with relative to end slices")
        return n + s.start

    if isinstance(sel, slice):
        return offset(sel, shape)

    if shape is None:
        return tuple(offset(s, None) for s in sel)

    return tuple(offset(s, n) for s, n in zip(sel, shape))


def offset_slice(sel, offset):
    def offset_slice_1d(sel, offset):
        start = sel.start + offset if sel.start is not None else offset
        stop = sel.stop + offset if sel.stop is not None else None
        return slice(start, stop, sel.step)

    if isinstance(sel, slice):
        return offset_slice_1d(sel, offset)

    return tuple(offset_slice_1d(s, o) for s, o in zip(sel, offset))


def array_memsize(shape, dtype):
    dtype = np.dtype(dtype)
    return functools.reduce(operator.mul, shape)*dtype.itemsize


def block_iterator(chunk_shape, roi, full_shape=None):
    def chunk_iterator_1d(chunk_sz, start, stop):
        pos = start

        off = pos % chunk_sz
        if off > 0:
            out = start + (chunk_sz - off)
            if out > stop:
                out = stop
            yield slice(pos, out)
            pos = out

        while pos + chunk_sz <= stop:
            yield slice(pos, pos + chunk_sz)
            pos = pos + chunk_sz

        if pos < stop:
            yield slice(pos, stop)

    if full_shape is not None:
        roi = tuple(norm_slice(s, n) for s, n in zip(roi, full_shape))
    else:
        roi = tuple(norm_slice(s) for s in roi)

    return itertools.product(*[chunk_iterator_1d(ch, sel.start, sel.stop)
                               for ch, sel in zip(chunk_shape, roi)])


class SharedBufferView(object):
    def __init__(self, buf):
        self._buf = buf

    @property
    def buffer(self):
        return self._buf

    def asarray(self, offset, shape, dtype, slot_sz=None):
        n_bytes = array_memsize(shape, dtype)

        if slot_sz is not None:
            if n_bytes > slot_sz:
                return None

        if offset + n_bytes > len(self._buf):
            return None

        return np.ndarray(shape,
                          dtype=dtype,
                          buffer=self._buf,
                          offset=offset)


class ChunkStoreAlloc(object):
    class Slot(object):
        def __init__(self, id, offset, ondone):
            self._id = id
            self._offset = offset
            self._ondone = ondone

        def __del__(self):
            if self._ondone:
                self._ondone(self._id)

        @property
        def id(self):
            return self._id

        @property
        def offset(self):
            return self._offset

        def release(self):
            if self._ondone:
                self._ondone(self._id)
                self._ondone = None
                self._id = None

    def __init__(self, chunk_sz, buf):
        self._view = SharedBufferView(buf)
        self._chunk_sz = chunk_sz
        self._n = len(buf)//chunk_sz
        self._free = [i for i in range(self._n)]

    @property
    def nfree(self):
        return len(self._free)

    @property
    def total(self):
        return self._n

    def alloc(self):
        try:
            idx = self._free.pop()
        except IndexError:
            return None

        return ChunkStoreAlloc.Slot(idx,
                                    idx*self._chunk_sz,
                                    lambda x: self._free.append(x))

    def check_size(self, shape, dtype):
        return array_memsize(shape, dtype) <= self._chunk_sz

    def asarray(self, slot, shape, dtype):
        """ slot is Int|Slot
        """
        if self.check_size(shape, dtype) is False:  # Slot memory is too small
            return None

        if isinstance(slot, (self.Slot,)):
            slot = slot._id

        if slot >= self._n:  # Not a valid slot id
            return None

        offset = slot*self._chunk_sz
        return self._view.asarray(offset, shape, dtype)


class Timer(object):
    def __init__(self, verbose=None, message=None):
        from timeit import default_timer

        if message is None:
            message = 'Elapsed time'
        elif verbose is None:
            verbose = True

        self.verbose = verbose
        self.message = message
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed = end - self.start
        self.elapsed_ms = self.elapsed * 1000
        if self.verbose:
            print('%s: %f ms' %
                  (self.message, self.elapsed_ms))


def test_shape_from_slice():
    import numpy as np

    assert shape_from_slice(np.s_[:], 10) == (10,)
    assert shape_from_slice(np.s_[:], (10,)) == (10,)

    assert shape_from_slice(np.s_[:, :], (10, 3)) == (10, 3)
    assert shape_from_slice(np.s_[:, :2], (10, 3)) == (10, 2)
    assert shape_from_slice(np.s_[:, 1:2], (10, 3)) == (10, 1)
    assert shape_from_slice(np.s_[::2, 1:2:3], (10, 3)) == (5, 1)
    assert shape_from_slice(np.s_[::3, 1:2:3], (10, 3)) == (4, 1)

    assert shape_from_slice(np.s_[0, :], (10, 3)) == (1, 3)

    assert shape_from_slice(np.s_[:3]) == (3,)
    assert shape_from_slice(np.s_[:3, 1:20:2]) == (3, 10)

    assert shape_from_slice(np.s_[-3:-1], (10,)) == (2,)
    # TODO: make this one work:  assert shape_from_slice(np.s_[-3:-1]) == (2,)


def test_offset_slice():
    s_ = np.s_
    assert offset_from_slice(s_[1:3]) == 1
    assert offset_from_slice(s_[1:3, :4, 5:]) == (1, 0, 5)
    assert offset_from_slice(s_[1:3, :4, -3:], (100, 200, 10)) == (1, 0, 7)

    assert offset_slice(s_[1:3], 10) == s_[11:13]
    assert offset_slice(s_[1:], 10) == s_[11:]
    assert offset_slice(s_[:], 1) == s_[1:]

    assert offset_slice(s_[1:3, 3:4], (10, 100)) == s_[11:13, 103:104]
    assert offset_slice(s_[1:3, 3:4, :5], (10, 100, 1000)) == s_[11:13, 103:104, 1000:1005]


def test_slot_alloc():
    buf = bytearray(4096*10)

    store = ChunkStoreAlloc(4096, buf)

    print(store.nfree, store.total)

    assert store.nfree == 10
    assert store.total == 10

    slot = store.alloc()
    assert slot is not None
    assert store.nfree == store.total - 1

    a = store.asarray(slot, (2, 3), 'uint8')
    b = store.asarray(slot.id, (2, 3), 'uint8')

    print(a)
    a[:, :] = 127
    print(b)

    assert (a == b).all()

    slot.release()
    assert store.nfree == store.total
