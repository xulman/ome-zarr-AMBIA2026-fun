import ngff_zarr as nz
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
)
import dask.array as da

shape = (10,64,128,128)
chunks = (1,64,64,64)
scale_factors = [2,4]
file_path = 'data/example_Tx1x2x2.ome.zarr'


def print_help():
    print("How to run this inside Python interpreter:")
    print('import step1_time_create_empty_skeleton as S1')
    print('import step2_time_contribute_chunk as S2')
    print('import step3_time_add_pyramids as S3')
    print('S1.main()')
    print('S2.main()')
    print('S3.main()')



def main():
    # skeleton empty array, that's lazily filled, so no 0.0f values/pixels do exist yet
    empty_initial_image = da.zeros(shape, dtype='float32', chunks=chunks)
    image = nz.to_ngff_image(empty_initial_image, dims=["t", "z", "y", "x"])

    multiscales = nz.to_multiscales(image, scale_factors=scale_factors, chunks=chunks, cache=False)

    # NGFF v0.6+ demo/intermezzo...
    # An affine that maps the intrinsic pixel system to some "output" system.
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

    # commit the coord system and transform into the metadata
    multiscales.metadata.coordinateSystems.append(output_cs)
    multiscales.metadata.coordinateTransformations = [affine]

    # final write to a drive/store
    nz.to_ome_zarr(file_path, multiscales, version="0.6", metadata_only=True)
    # NB: if metadata_only was omitted, it still produces "only the jsons"
    print(f"written {file_path}")


if __name__ == "__main__":
    main()
