import numpy as np
import ngff_zarr as nz
import step1_time_create_empty_skeleton as S1


def write_whole_timepoint(time_index: int, base_value = 100):
    # read the skeleton/reference OME-Zarr, and get path to the base-level image
    multiscales = nz.from_ngff_zarr(S1.file_path)

    # read the daskarray facade around the image array
    # (the image is initially an empty skeleton, but it gets
    #  filled progressively with a code like here below),
    # and do this for every pyramid level
    paths = [ ds.path for ds in multiscales.metadata.datasets ]
    images = [ nz.open_array(S1.file_path, path) for path in paths ]

    # plan B
    # example: write progressively full time point into a buffer at base/full-resolution
    buf = np.zeros(images[0].shape[1:], dtype=images[0].dtype)
    # ... fill buf progressively as data arrives ...
    buf[ 0:32,:,:] = base_value
    buf[32:64,:,:] = base_value+1

    # turn 'buf' into a "one-time" multiscales (that would be copied into the original dataset)
    base_image = nz.to_ngff_image(buf, dims=multiscales.metadata.dimension_names[1:])
    tmp_multiscales = nz.to_multiscales(base_image, scale_factors=S1.scale_factors, chunks=images[0].chunks[1:])
    # NB: scale_factors are in OME-Zarr nowhere stored explicitly, we could either compute them,
    #     or just pull them from the memory we had used when creating the skeleton dataset
    # NB: this solution uses ngff-zarr's internal code to produce the pyramids,
    #     not a code of my own (which could compute pyramids that deviate from those from ngff-zarr)

    # publish: one full time point
    # (...benefiting that the write to a store happens immediately)
    for level in range(len(images)):
        images[level][time_index] = tmp_multiscales.images[level].data


def main():
    write_whole_timepoint(1, 30)
    write_whole_timepoint(3, 50)
    write_whole_timepoint(4, 60)


if __name__ == "__main__":
    main()
