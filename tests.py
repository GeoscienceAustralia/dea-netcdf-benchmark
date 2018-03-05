from pathlib import Path
import numpy as np

from mpnetcdf4.utils import (NamedObjectCache,
                             Timer)

from mpnetcdf4.ncread import (NetcdfProcProxy,
                              ReaderFactory,
                              eh5_open,
                              eh5_close,
                              ExternalNetcdfReader,
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


def test_netdcf_proxy():
    fname = TEST_HDF_FILE

    for i in range(3):
        with Timer('Setup'):
            st = SharedState(mb=1)
            proc = st.make_procs(1)[0]

    for i in range(5):
        with Timer(message='Open'):
            f = NetcdfProcProxy(fname, proc)

        if i == 0:
            print(list(f.info.bands))

        future = f.read_to_shared('red', np.s_[0, :200, :200], 0)
        assert future.result() is True
        f.close()


def test_parallel_open():
    fname = TEST_HDF_FILE
    nprocs = 8

    with Timer(message='Prepare'):
        st = SharedState()
        pp = st.make_procs(nprocs)

    for i in range(4):
        with Timer(message='Open x%d' % nprocs):
            ff = NetcdfProcProxy.parallel_open(fname, pp)

        if i == 0:
            print(list(ff[0].info.bands))

        del ff


def test_reader_factory():
    fname = TEST_HDF_FILE

    mpr = ReaderFactory(2)
    f = mpr.open(fname)
    assert f is not None
    f.close()

    with mpr.open(fname) as f:
        xx = f.read()
        print(xx)
        assert xx is not None


if __name__ == '__main__':
    pass
