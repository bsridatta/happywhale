from re import I
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List, Optional
import pandas as pd
import albumentations
from skimage import io
from skimage.color import gray2rgb


class Whales(Dataset):
    def __init__(
        self,
        data_root: str = f"{os.environ['HOME']}/lab/data/",
        folds: Optional[List[int]] = None,
        no_augment: bool = False,
        img_size: int = 512,
        image_path: str = "train_images/",
        csv_path: str = "train_equal_species_ids.csv",
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_path = self.data_root + image_path
        self.metadata = pd.read_csv(data_root + csv_path)

        # select folds - for train/valid
        if folds:
            self.metadata: pd.DataFrame = self.metadata[
                self.metadata.k_fold.isin(folds)
            ].reset_index(drop=True)

        # TODO: for cropped detic dataset, something wrong make sure to sure the right dataset
        # self.metadata["image"] = self.metadata["image"].apply(
        #     lambda x: x.replace("jpg", "jpeg")
        # )

        # to allow test without full dataset
        self.metadata = self.metadata[
            self.metadata["image"].isin(os.listdir(self.image_path))
        ]

        print(f"[INFO]: Dataset size - {len(self.metadata)}")

        # image augmentations
        # no_augment, just pre-processing for validation and test
        augs = []
        if img_size != 512:
            augs.append(albumentations.Resize(img_size, img_size, always_apply=True))

        if no_augment:
            augs.append(
                albumentations.Normalize(always_apply=True),
            )
        # else apply augmentations for train
        else:
            augs.extend(
                [
                    albumentations.ShiftScaleRotate(rotate_limit=40, p=0.9),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        self.augmentations = albumentations.Compose(augs)

    def __len__(self):
        return len(self.metadata.image)

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        sample = {"image_id": data.image}

        # just to exclude test csv which has no meta data
        if "cat_id" in data and "species" in data:
            sample["individual_id"] = data.cat_id
            sample["individual_id_org"] = data.individual_id
            sample["species"] = data.species

        image = self.read_image(data.image)
        image = self.augmentations(image=image)["image"]
        image = torch.tensor(image, dtype=torch.float32)
        sample["image"] = image.permute(2, 0, 1)

        return sample

    def read_image(self, image_id: str) -> np.ndarray:
        image = io.imread(self.image_path + image_id)
        return image if len(image.shape) == 3 else gray2rgb(image)


if __name__ == "__main__":
    dataset = Whales(folds=[0, 1, 2, 3])
    print(dataset[1]["image"].shape)
    print(dataset[1].keys())

    dataset = Whales(folds=[4], no_augment=True)
    print(dataset[0]["image"].shape)
    print(dataset[0].keys())
    print(dataset[0]["individual_id_org"])

    # dataset = Whales(folds=[], no_augment=True)
    # print(dataset[1]["image"].shape)
    # print(dataset[1].keys())

    # dataset = Whales(
    #     folds=[],
    #     no_augment=True,
    #     csv_path="sample_submission.csv",
    #     image_path="test_images/",
    # )
    # print(dataset[1]["image"].shape)
    # print(dataset[1].keys())

    # dataset = Whales(
    #     folds=[],
    #     no_augment=True,
    #     csv_path="sample_submission.csv",
    #     image_path="test_images/",
    # )
    # print(dataset[1]["image"].shape)
    # print(dataset[1].keys())
