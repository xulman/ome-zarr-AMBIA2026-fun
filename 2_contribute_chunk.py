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


multiscales = nz.from_ngff_zarr('example_1x2x2.ome.zarr', validate=False)
path = multiscales.metadata.datasets[0].path

image = nz.open_array('example_1x2x2.ome.zarr', path)

# example: one chunk write
# beware: writes to the drive immediatelly!
image[0:64,0:64,0:64] = 99.1

# plan B, write progressively first into a buffer
buf = np.zeros(image.chunks, dtype=image.dtype)   # one chunk's worth
# ... fill buf progressively as data arrives ...
buf[ 0:32,:,:] = 128.1
buf[32:64,:,:] = 182.1
# publish: one full-chunk, chunk-aligned write → single encode, single write
image[0:64,0:64,0:64] = buf



