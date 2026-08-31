import numpy as np
import ngff_zarr as nz
import spatial_step1_create_empty_skeleton as S1

def lbnd(index_vector, chunk_size, axis_index):
    return index_vector[axis_index] * chunk_size[axis_index]

def rbnd(index_vector, chunk_size, axis_index):
    return (index_vector[axis_index] +1) * chunk_size[axis_index]


def write_3D_chunk_at_index(index_vector: list[int], base_value = 100):
    # read the skeleton/reference OME-Zarr, and get path to the base-level image
    multiscales = nz.from_ngff_zarr(S1.file_path, validate=False)
    path = multiscales.metadata.datasets[0].path

    # read the daskarray facade around the image array
    # (the image is initially an empty skeleton, but it gets
    #  filled progressively with a code like here below)
    image = nz.open_array(S1.file_path, path)

    # shortcuts
    bl = lambda idx : lbnd(index_vector, image.chunks, idx)
    br = lambda idx : rbnd(index_vector, image.chunks, idx)

    # plan A:
    # example: one chunk write
    # beware: writes to the drive immediately!
    # image[bl(0):br(0), bl(1):br(1), bl(2):br(2)] = base_value
    # enable the above and check here the store content....

    # plan B
    # example: write progressively first into a buffer
    buf = np.zeros(image.chunks, dtype=image.dtype)   # one chunk's worth
    # ... fill buf progressively as data arrives ...
    buf[ 0:32,:,:] = base_value
    buf[32:64,:,:] = base_value+1
    # publish: one full-chunk, chunk-aligned write → single encode, single write
    # (...benefiting that the write to a store happens immediately)
    image[bl(0):br(0), bl(1):br(1), bl(2):br(2)] = buf


def main():
    write_3D_chunk_at_index([0,0,0], 30)
    write_3D_chunk_at_index([0,1,0], 50)
    write_3D_chunk_at_index([0,0,1], 70)


if __name__ == "__main__":
    main()
