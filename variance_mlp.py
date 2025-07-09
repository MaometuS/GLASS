from torch import nn
import torch.nn.functional as F
import torch
import math
import common
import click
from model import PatchMaker
import utils
import backbones

def downsample(x: torch.Tensor) -> torch.Tensor:
    H = W = int(x.shape[0]**0.5)
    x = x.view(H, W, x.shape[1])

    # Downsample by 2x using average pooling
    x = x.permute(2, 0, 1).unsqueeze(0)
    x_down = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

    # Flatten back
    x_down = x_down.squeeze(0).permute(1, 2, 0).reshape(-1, x.shape[1])
    return x_down

class VarianceMLP(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=1024):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.variance_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Softplus()  # ensure positive variance
        )

    def forward(self, x):  # x: [B, 1296, 1536]
        x = self.token_mlp(x)     # [B, 1296, 1536]
        x = x.mean(dim=1)         # [B, 1536] — aggregate over patches
        var = self.variance_head(x)  # [B, 1536]
        return var

class Embedder:
    def load(
            self,
            backbone,
            layers_to_extract_from,
            input_shape,
            device,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, False
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)


    def embed(self, images):
        """Returns feature embeddings for images."""
        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("aug_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=8, type=int)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
):
    _DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"], "visa": ["datasets.visa", "VisADataset"],
                 "mpdd": ["datasets.mvtec", "MVTecDataset"], "wfdd": ["datasets.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader.name = get_name + "_" + subdataset

            if test == 'ckpt':
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                train_dataloader.name = test_dataloader.name
            else:
                train_dataloader = test_dataloader

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders

@main.command("net")
@click.option("--backbone", "-b", type=str, default="wideresnet50")
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
def net(
        backbone,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
):

    backbone_model = backbones.load(backbone)
    backbone_model.name, backbone_model.seed = backbone, None

    def get_embedder(input_shape, device) -> Embedder:

        embedder = Embedder()
        embedder.load(
            backbone_model,
            layers_to_extract_from,
            input_shape,
            device,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize,
        )
        return embedder

    return "get_embedder", get_embedder

@main.result_callback()
def run(
        methods,
        results_path,
        log_project,
        log_group,
        run_name,
        seed,
        test,
        gpu,
):
    methods = {key: item for (key, item) in methods}

    # run_save_path = utils.create_storage_folder(
    #     results_path, log_project, log_group, run_name, mode="overwrite"
    # )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed)
        dataset_name = dataloaders["training"]
        print("Dataset is: ", dataset_name)
        imagesize = dataloaders["training"].dataset.imagesize
        dataset = dataloaders["training"].dataset

        embedder: Embedder = methods["get_embedder"](imagesize, device)

        for data in dataloaders["training"]:
            embedding = embedder.embed(data["image"].to(device))
            print("The embedding shape is: " + str(embedding.shape))

if __name__ == "__main__":
    main()
