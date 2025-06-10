import coinpp.conversion as conversion
import coinpp.models as models

import data.kitti as kitti
import torchvision
import yaml
from pathlib import Path


def get_dataset_root(dataset_name: str):
    """Returns path to data based on dataset_paths.yaml file."""
    with open(r"data/dataset_paths.yaml") as f:
        dataset_paths = yaml.safe_load(f)

    return Path(dataset_paths[dataset_name])


def dataset_name_to_dims(dataset_name):
    """Returns appropriate dim_in and dim_out for dataset."""
    if dataset_name == "mnist":
        dim_in, dim_out = 2, 1
    if dataset_name in ("cifar10", "kodak", "vimeo90k"):
        dim_in, dim_out = 2, 3
    if dataset_name == "fastmri":
        dim_in, dim_out = 3, 1
    if dataset_name == "kitti":
        dim_in, dim_out = 3, 1
    if dataset_name == "era5":
        dim_in = 3
        dim_out = 1
    if dataset_name == "librispeech":
        dim_in = 1
        dim_out = 1
    return dim_in, dim_out


def get_datasets_and_converter(args, force_no_random_crop=False):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
        force_no_random_crop (bool): If True, forces datasets to not use random
            crops (which is the default for the training set when using
            patching). This is useful after the model is trained when we store
            modulations.
    """
    # Extract input and output dimensions of function rep
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset)

    # When using patching, perform random crops equal to patch size on training
    # dataset
    use_patching = hasattr(args, "patch_shape") and args.patch_shape != [-1]
    if use_patching:
        if dim_in == 2:
            random_crop = torchvision.transforms.RandomCrop(args.patch_shape)


    if "kitti" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("kitti")

        if args.train_dataset == "kitti":
            train_dataset = kitti.KITTI(
                root='./data/kitti',
                split="train",
            )

        if args.test_dataset == "kitti":
            test_dataset = kitti.KITTI(
                root='./data/kitti',
                split="train",
            )


    return train_dataset, test_dataset, converter


def get_model(args):
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset)
    return models.ModulatedSiren(
        dim_in=dim_in,
        dim_hidden=args.dim_hidden,
        dim_out=dim_out,
        num_layers=args.num_layers,
        w0=args.w0,
        w0_initial=args.w0,
        modulate_scale=args.modulate_scale,
        modulate_shift=args.modulate_shift,
        use_latent=args.use_latent,
        latent_dim=args.latent_dim,
        modulation_net_dim_hidden=args.modulation_net_dim_hidden,
        modulation_net_num_layers=args.modulation_net_num_layers,
    ).to(args.device)
