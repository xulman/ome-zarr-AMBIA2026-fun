import ngff_zarr as nz
import step1_create_empty_skeleton as S1


def main():
    # read the skeleton/reference OME-Zarr, and its base image fully
    multiscales = nz.from_ngff_zarr(S1.file_path)
    base_image = multiscales.images[0]

    # re-do the metadata again, notice that it is basically the same params like in S1,
    # except that more scale factors are added; similarly below for the to_ome_zarr()
    multiscales = nz.to_multiscales(base_image, scale_factors=[2,4], chunks=S1.chunks)
    nz.to_ome_zarr(S1.file_path, multiscales, version="0.6", overwrite=False, start_level=1)


if __name__ == "__main__":
    main()
