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

empty_initial_image = da.zeros(shape, dtype='float32', chunks=chunks)
image = nz.to_ngff_image(empty_initial_image, dims=["z", "y", "x"])

multiscales = nz.to_multiscales(image, scale_factors=[], chunks=chunks, cache=False)

# An affine that maps the intrinsic pixel system to an "output" system.
output_cs = CoordinateSystem(
    name="output",
    axes=[Axis(name=d, type="space") for d in ("z", "y", "x")],
)

affine = Affine(
    affine=[
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 1.0, 15.0],
    ],
    input=CoordinateSystemIdentifier(
        name=multiscales.metadata.intrinsic_coordinate_system.name
    ),
    output=CoordinateSystemIdentifier(name=output_cs.name),
    name="to_output",
)

multiscales.metadata.coordinateSystems.append(output_cs)
multiscales.metadata.coordinateTransformations = [affine]

nz.to_ome_zarr("example_1x2x2.ome.zarr", multiscales, version="0.6")
