"""
Datacube interop functions are here
"""
import numpy as np
from itertools import chain
from types import SimpleNamespace
from datacube.storage.storage import measurement_paths
from datacube.utils import uri_to_local_path
from datacube.api import GridWorkflow


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


def first_val(x):
    return next(iter(x.values()))


def list_native_cell(product, cell_index, dc, **query):
    index = dc.index

    p = index.products.get_by_name(product)
    if p.grid_spec is None:
        raise ValueError('Supplied product does not have a grid spec')

    gw = GridWorkflow(index, grid_spec=p.grid_spec)
    tile = gw.list_cells(cell_index=cell_index,
                         product=product,
                         **query)[cell_index]

    return list(flatmap(lambda x: x, tile.sources.values))


def group_by_storage(dss, bands=None):
    """

    returns [StorageResource]

    StorageResource
      .uri           - string, URI of the resource
      .local_path    - PosixPath, path on a filesystem, could be None if not a file resource
      .bands         - Dictionary of bands (copied from Dataset)
      .time          - np.ndarray<datetime64[ns]> Timestamps to be read from this resource
      .datasets      - List<datacube.Dataset> referencing this resource
    """
    su_all = {}

    if bands is None:
        def check_band(band):
            return True
    else:
        bands = set(bands)

        def check_band(band):
            return band in bands

    def local_path(uri):
        try:
            return uri_to_local_path(uri)
        except ValueError:
            return None

    def update(su, ds, band=None):
        if band is None:
            bb = {k: ds.measurements[k]
                  for k in ds.measurements if check_band(k)}
        else:
            bb = {band: ds.measurements[band]}

        if su not in su_all:
            su_all[su] = SimpleNamespace(bands=bb,
                                         uri=su,
                                         local_path=local_path(su),
                                         datasets=[ds])
        else:
            su_all[su].datasets.append(ds)

    for ds in dss:
        pp = measurement_paths(ds)
        paths = set(pp.values())

        if len(paths) == 1:  # All bands in one file
            update(paths.pop(), ds)
        elif len(paths) == len(pp):  # Each band in it's own file
            for band, file in pp.items():
                if check_band(band):
                    update(file, ds, band)
        else:
            raise ValueError('Not supporting multiple multi-band files')

    for s in su_all.values():
        s.time = np.array([ds.center_time for ds in s.datasets], dtype='datetime64[ns]')

    return sorted(su_all.values(), key=lambda s: s.time[0])


def compute_time_slice(requested_time, file_time):
    """
    Given requested time stamps and available timestamps (both assumed to be
    sorted in ascending order), computes roi such that

    requested_time in file_time[roi]


    Returns (roi, contigous, complete)

    Where:
      roi:        slice object
      contigous:  True|False if False not all file stamps in the range are needed
      complete:   True|False, if False some requested timestamps were not found
    """
    assert requested_time.dtype == file_time.dtype

    ii = np.where((file_time >= requested_time.min()) * (file_time <= requested_time.max()))[0]

    if len(ii) == 0:
        raise ValueError("No overlap")

    roi = slice(ii[0], ii[-1]+1)

    file_time = set(file_time[roi])
    requested_time = set(requested_time)

    contigous = (file_time == requested_time)
    complete = requested_time.issubset(file_time)

    return roi, contigous, complete
