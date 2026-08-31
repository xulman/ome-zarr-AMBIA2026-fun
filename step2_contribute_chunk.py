import numpy as np
import ngff_zarr as nz
import step1_create_empty_skeleton as S1


def main():
    # read the skeleton/reference OME-Zarr, and get path to the base-level image
    multiscales = nz.from_ngff_zarr(S1.file_path, validate=False)
    path = multiscales.metadata.datasets[0].path

    # read the daskarray facade around the image array
    # (the image is initially an empty skeleton, but it gets
    #  filled progressively with a code like here below)
    image = nz.open_array(S1.file_path, path)

    # plan A:
    # example: one chunk write
    # beware: writes to the drive immediately!
    image[0:64,0:64,0:64] = 99.1

    # check here the store content....

    # plan B
    # example: write progressively first into a buffer
    buf = np.zeros(image.chunks, dtype=image.dtype)   # one chunk's worth
    # ... fill buf progressively as data arrives ...
    buf[ 0:32,:,:] = 128.1
    buf[32:64,:,:] = 182.1
    # publish: one full-chunk, chunk-aligned write → single encode, single write
    # (...benefiting that the write to a store happens immediately)
    image[0:64,0:64,0:64] = buf


if __name__ == "__main__":
    main()
