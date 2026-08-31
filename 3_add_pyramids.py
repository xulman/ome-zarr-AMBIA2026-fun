import numpy as np
import ngff_zarr as nz
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
)
import dask.array as da

shape = (64,128,128)
chunks = (64,64,64)


multiscales = nz.from_ngff_zarr('example_1x2x2.ome.zarr')
base_image = multiscales.images[0]
multiscales = nz.to_multiscales(base_image, scale_factors=[2,4], chunks=chunks)
nz.to_ome_zarr("example_1x2x2.ome.zarr", multiscales, version="0.6", overwrite=False, start_level=1)
