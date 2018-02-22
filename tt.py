from pathlib import Path
import numpy as np
import yaml
from attrdict import AttrDict

from utils import (NamedObjectCache,
                   Timer,
                   block_iterator,
                   shape_from_slice)

from ncread import (eh5_open, eh5_close, eh5_info, eh5_read_to_shared,
                    NetcdfProcProxy,
                    ExternalNetcdfReader)

TEST_HDF_FILE = ('/g/data2/rs0/datacube/002/LS8_OLI_NBAR/4_-35/'
                 'LS8_OLI_NBAR_3577_4_-35_20171107004524000000_v1513646960.nc')

TEST_HDF_STACKED_FILE = ('/g/data2/rs0/datacube/002/LS8_OLI_NBAR/4_-35/'
                         'LS8_OLI_NBAR_3577_4_-35_2015_v1496400956.nc')


if not Path(TEST_HDF_FILE).exists():
    TEST_HDF_FILE = './sample.nc'


def test_1():
    fname = TEST_HDF_FILE
    fd1 = eh5_open(fname)
    assert fd1 > 0
    fd2 = eh5_open(fname)
    print(fd1, fd2)
    assert fd1 == fd2

    assert eh5_close(fd1) is True
    assert eh5_close(fd1) is False


def test_external_eh5_pp_shared():
    fname = TEST_HDF_FILE

    pp = ExternalNetcdfReader.mk_proc()

    fd = pp.submit(eh5_open, fname)
    fd = fd.result()

    assert fd > 0

    info = pp.submit(eh5_info, fd)
    info = info.result()
    print(yaml.dump(info))

    varname = 'blue'
    ii = AttrDict(info['vars'][varname])

    slot_alloc, _ = ExternalNetcdfReader.slot_allocator(ii.chunks, ii.dtype)
    assert slot_alloc is not None

    (slot, my_view) = slot_alloc()

    assert slot is not None

    roi = np.s_[0, 1000:1200, 2000:2200]
    r = pp.submit(eh5_read_to_shared, fd, varname, roi, slot.offset)
    print(r.result())
    print(my_view)
    slot.release()


def test_slot_alloc():
    slot_alloc, _ = ExternalNetcdfReader.slot_allocator((1, 200, 200), 'int16')
    assert slot_alloc is not None

    (s1, a1) = slot_alloc()
    (s2, a2) = slot_alloc()
    print(s1.id, a1)
    print(s2.id, a2)


def test_named_cache():
    fname = TEST_HDF_FILE

    for i in range(3):
        fd, obj = NamedObjectCache(ExternalNetcdfReader)(fname)
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
                      proc=None,
                      chunk_scale=None):
    import concurrent.futures

    f = NetcdfProcProxy(fname, proc=proc)

    if dump_info:
        print(yaml.dump(f.info))

    ii = AttrDict(f.info['vars'][measurement])

    if chunk_scale is None:
        read_chunk = ii.chunks
    else:
        read_chunk = tuple(ch*s for ch, s in zip(ii.chunks, chunk_scale))

    slot_alloc, _ = ExternalNetcdfReader.slot_allocator(read_chunk, ii.dtype)
    assert slot_alloc is not None

    scheduled = set()

    dst = np.zeros(ii.shape, dtype=ii.dtype)
    ROI = np.s_[:, :, :]

    def finalise(r):
        slot, roi, my_view = r._userdata
        if r.result():
            dst[roi] = my_view
        slot.release()

    for roi in block_iterator(read_chunk, ROI, ii.shape):
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
    return dst, f.proc


def test_read_via_external():
    print('\nStarting read test')

    with Timer(message='Prepare'):
        proc = ExternalNetcdfReader.mk_proc()

    for i, band in enumerate('red green blue nir'.split(' ')):
        with Timer(message='Read::%s 2x2 (%d)' % (band, i)):
            dd, proc = read_via_external(TEST_HDF_FILE, band,
                                         chunk_scale=(1, 2, 2),
                                         proc=proc)

            assert dd.shape == (1, 4000, 4000)
            print(dd.shape, dd.dtype)


def read_via_external_mp(fname,
                         measurement,
                         dump_info=False,
                         procs=None,
                         chunk_scale=None):
    import concurrent.futures

    if procs is None:
        procs = 1

    if isinstance(procs, int):
        procs = [ExternalNetcdfReader.mk_proc() for _ in range(procs)]

    ff = []
    info = None
    for i, proc in enumerate(procs):
        f = NetcdfProcProxy(fname, proc=proc, info=info)
        ff.append(f)

        if i == 0:
            info = f.info

    if dump_info:
        print(yaml.dump(info))

    ii = AttrDict(info['vars'][measurement])

    if chunk_scale is None:
        read_chunk = ii.chunks
    else:
        read_chunk = tuple(ch*s for ch, s in zip(ii.chunks, chunk_scale))

    slot_alloc, _ = ExternalNetcdfReader.slot_allocator(read_chunk, ii.dtype)
    assert slot_alloc is not None

    scheduled = set()

    dst = np.zeros(ii.shape, dtype=ii.dtype)
    ROI = np.s_[:, :, :]

    f_idx = -1

    def choose_target():
        '''For now just round-robin between ff'''
        nonlocal f_idx

        f_idx = f_idx + 1
        if f_idx >= len(ff):
            f_idx = 0
        return ff[f_idx]

    def finalise(r):
        slot, roi, my_view = r._userdata
        if r.result():
            dst[roi] = my_view
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

    for roi in block_iterator(read_chunk, ROI, ii.shape):
        (slot, my_view) = get_slot(shape_from_slice(roi))
        f = choose_target()
        future = f.read_to_shared(measurement, roi, slot.offset)
        future._userdata = (slot, roi, my_view)
        scheduled.add(future)

    xx = concurrent.futures.wait(scheduled)
    for r in xx.done:
        finalise(r)

    for f in ff:
        f.close()

    return dst, procs


def test_read_via_external_mp(nprocs=2, chunk_scale=(1, 2, 2)):
    print('\nStarting concurrent read test: %d' % (nprocs,))

    procs = nprocs

    for i, band in enumerate('red green blue nir'.split(' ')):
        with Timer(message='Read(x%s)::%s %dx%d (%d)' % (nprocs, band,
                                                         chunk_scale[1],
                                                         chunk_scale[2],
                                                         i)):
            dd, procs = read_via_external_mp(TEST_HDF_FILE, band,
                                             chunk_scale=(1, 2, 2),
                                             procs=procs)

            assert dd.shape == (1, 4000, 4000)
            print(dd.shape, dd.dtype)


if __name__ == '__main__':
    ExternalNetcdfReader.static_init(100*(1 << 20))
    test_read_via_external()
    test_read_via_external_mp(2)
    test_read_via_external_mp(4)
    test_read_via_external_mp(8)
