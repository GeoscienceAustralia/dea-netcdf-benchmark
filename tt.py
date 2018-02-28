from pathlib import Path
import numpy as np
import yaml

from utils import (NamedObjectCache,
                   Timer,
                   block_iterator,
                   dst_from_src,
                   select_all,
                   shape_from_slice)

from ncread import (NetcdfProcProxy,
                    ExternalNetcdfReader,
                    RoundRobinSelector,
                    SharedState)

TEST_HDF_FILE = ('/g/data2/rs0/datacube/002/LS8_OLI_NBAR/4_-35/'
                 'LS8_OLI_NBAR_3577_4_-35_20171107004524000000_v1513646960.nc')

TEST_HDF_STACKED_FILE = ('/g/data2/rs0/datacube/002/LS8_OLI_NBAR/4_-35/'
                         'LS8_OLI_NBAR_3577_4_-35_2015_v1496400956.nc')


if not Path(TEST_HDF_FILE).exists():
    TEST_HDF_FILE = './sample.nc'

if not Path(TEST_HDF_STACKED_FILE).exists():
    TEST_HDF_STACKED_FILE = './sample_stacked.nc'


def test_1():
    from ncread import eh5_open, eh5_close
    st = SharedState(mb=1)

    fname = TEST_HDF_FILE
    fd1 = eh5_open(fname, st.view)
    assert fd1 > 0
    fd2 = eh5_open(fname, st.view)
    print(fd1, fd2)
    assert fd1 == fd2

    assert eh5_close(fd1) is True
    assert eh5_close(fd1) is False


def test_slot_alloc():
    st = SharedState()
    slot_alloc, _ = st.slot_allocator((1, 200, 200), 'int16')
    assert slot_alloc is not None

    (s1, a1) = slot_alloc()
    (s2, a2) = slot_alloc()
    print(s1.id, a1)
    print(s2.id, a2)


def test_named_cache():
    fname = TEST_HDF_FILE
    st = SharedState()
    view = st.view

    for i in range(3):
        fd, obj = NamedObjectCache(lambda fname: ExternalNetcdfReader(fname, view))(fname)
        print('FD:', fd, obj)
        obj = NamedObjectCache(None).lookup(fd)
        print('...', obj)

    obj = NamedObjectCache().clear_by_name(fname)
    print('Clear:', obj)
    assert obj is not None

    obj = NamedObjectCache().lookup(fd)
    assert obj is None
    print('Should be missing:', obj)


def read_via_external(fname,
                      measurement,
                      dump_info=False,
                      src_roi=None,
                      state=None,
                      proc=None,
                      chunk_scale=None):
    import concurrent.futures

    if state is None:
        state = SharedState()

    if proc is None:
        proc = state.make_procs(1)[0]

    f = NetcdfProcProxy(fname, proc=proc)

    if dump_info:
        print(yaml.dump(f.info))

    ii = f.info.bands[measurement]
    src_shape = ii.shape

    if src_roi is None:
        src_roi = select_all(src_shape)

    if chunk_scale is None:
        read_chunk = ii.chunks
    else:
        read_chunk = tuple(ch*s for ch, s in zip(ii.chunks, chunk_scale))

    slot_alloc, _ = state.slot_allocator(read_chunk, ii.dtype)
    assert slot_alloc is not None

    scheduled = set()

    dst_shape = shape_from_slice(src_roi, src_shape)
    dst_roi = dst_from_src(src_roi, src_shape)

    dst = np.zeros(dst_shape, dtype=ii.dtype)

    def finalise(r):
        slot, roi, my_view = r._userdata
        if r.result():
            dst[dst_roi(roi)] = my_view
        slot.release()

    for roi in block_iterator(read_chunk, src_roi, src_shape):
        (slot, my_view) = slot_alloc(shape_from_slice(roi))

        while slot is None:
            xx = concurrent.futures.wait(scheduled, return_when='FIRST_COMPLETED')
            for r in xx.done:
                finalise(r)
            scheduled = xx.not_done
            (slot, my_view) = slot_alloc(shape_from_slice(roi))

        future = f.read_to_shared(measurement, roi, slot.offset)
        future._userdata = (slot, roi, my_view)

        scheduled.add(future)

    xx = concurrent.futures.wait(scheduled)
    for r in xx.done:
        finalise(r)

    f.close()
    return dst, state, f.proc


def test_read_via_external():
    print('\nStarting read test')

    with Timer(message='Prepare'):
        state = SharedState()
        proc = state.make_procs(1)[0]

    for i, band in enumerate('red green blue nir'.split(' ')):
        with Timer(message='Read::%s 2x2 (%d)' % (band, i)):
            dd, state, proc = read_via_external(TEST_HDF_FILE, band,
                                                chunk_scale=(1, 2, 2),
                                                state=state, proc=proc)

            assert dd.shape == (1, 4000, 4000)
            print(dd.shape, dd.dtype)


def read_via_external_mp(fname,
                         measurement,
                         src_roi=None,
                         dump_info=False,
                         state=None,
                         procs=None,
                         chunk_scale=None):
    import concurrent.futures

    if state is None:
        state = SharedState()

    if procs is None:
        procs = 1

    if isinstance(procs, int):
        procs = state.make_procs(procs)

    ff = []
    info = None
    for i, proc in enumerate(procs):
        f = NetcdfProcProxy(fname, proc=proc, info=info)
        ff.append(f)

        if i == 0:
            info = f.info

    if dump_info:
        print(yaml.dump(info))

    ii = info.bands[measurement]
    src_shape = ii.shape

    if src_roi is None:
        src_roi = select_all(src_shape)

    if chunk_scale is None:
        read_chunk = ii.chunks
    else:
        read_chunk = tuple(ch*s for ch, s in zip(ii.chunks, chunk_scale))

    slot_alloc, _ = state.slot_allocator(read_chunk, ii.dtype)
    assert slot_alloc is not None

    scheduled = set()

    dst_shape = shape_from_slice(src_roi, src_shape)
    dst_roi = dst_from_src(src_roi, src_shape)

    dst = np.zeros(dst_shape, dtype=ii.dtype)

    read_to_shared = RoundRobinSelector([f.read_to_shared for f in ff])

    def finalise(r):
        slot, roi, my_view = r._userdata
        if r.result():
            dst[dst_roi(roi)] = my_view
        else:
            print('Failed one of the reads:', roi, slot.id)
        slot.release()

    def get_slot(shape):
        nonlocal scheduled
        (slot, my_view) = slot_alloc(shape)

        while slot is None:
            xx = concurrent.futures.wait(scheduled, return_when='FIRST_COMPLETED')
            for r in xx.done:
                finalise(r)
            scheduled = xx.not_done
            (slot, my_view) = slot_alloc(shape)

        return (slot, my_view)

    for roi in block_iterator(read_chunk, src_roi, src_shape):
        (slot, my_view) = get_slot(shape_from_slice(roi))
        future = read_to_shared(measurement, roi, slot.offset)
        future._userdata = (slot, roi, my_view)
        scheduled.add(future)

    xx = concurrent.futures.wait(scheduled)
    for r in xx.done:
        finalise(r)

    for f in ff:
        f.close()

    return dst, state, procs


def test_read_via_external_mp(nprocs=2,
                              fname=TEST_HDF_FILE,
                              measurements='red green blue nir'.split(' '),
                              chunk_scale=(1, 2, 2),
                              src_roi=None):
    print('\nStarting concurrent read test: %d\n %s' % (nprocs, fname))

    state = None
    procs = nprocs
    out = {}

    for i, band in enumerate(measurements):
        with Timer(message='Read(x%s)::%s %dx%d (%d)' % (nprocs, band,
                                                         chunk_scale[1],
                                                         chunk_scale[2],
                                                         i)):
            dd, state, procs = read_via_external_mp(fname, band,
                                                    src_roi=src_roi,
                                                    chunk_scale=(1, 2, 2),
                                                    state=state, procs=procs)

            print(dd.shape, dd.dtype)
            out[band] = dd
    return out


if __name__ == '__main__':
    test_read_via_external()
    test_read_via_external_mp(2)
    test_read_via_external_mp(4)
    test_read_via_external_mp(8)
