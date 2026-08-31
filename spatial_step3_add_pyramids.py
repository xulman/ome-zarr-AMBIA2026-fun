import ngff_zarr as nz
import dataclasses
import spatial_step1_create_empty_skeleton as S1


def main():
    # read the skeleton/reference OME-Zarr, and its base image fully
    ms_orig = nz.from_ngff_zarr(S1.file_path)
    chunks = ms_orig.images[0].data.chunksize

    # re-do the metadata again, by building a fresh new multiscales object that
    # pulls in the pyramids/scales and which will blend with the original metadata
    ms_built = nz.to_multiscales(ms_orig.images[0], scale_factors=[2, 4], chunks=chunks)

    start = 1  # keep levels < start from disk, append levels >= start
    merged_datasets = ms_orig.metadata.datasets[:start] + ms_built.metadata.datasets[start:]
    merged_metadata = dataclasses.replace(ms_orig.metadata, datasets=merged_datasets)
    ms_spliced = dataclasses.replace(ms_built, metadata=merged_metadata, root_attributes=ms_orig.root_attributes)

    nz.to_ome_zarr(S1.file_path, ms_spliced, version="0.6", overwrite=False, start_level=1)


if __name__ == "__main__":
    main()
