import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List
import pandas as pd
import albumentations
from skimage import io
from skimage.color import gray2rgb


class Whales(Dataset):
    def __init__(
        self,
        data_root: str = f"{os.environ['HOME']}/lab/data/",
        folds: List[int] = [0],
        is_train: bool = True,
        img_size: int = 500,
        train_image_path: str = "train_images/",
        test_image_path: str = "test_images/",  # TODO: just image path pass train or test in creation?
        train_csv_path: str = "train_equal_species_ids.csv",
        test_csv_path: str = "sample_submission.csv",
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.is_train = is_train
        self.image_path: str = ""

        # read metadata
        if is_train:  # train/valid
            self.metadata = self.get_train_csv(data_root + train_csv_path)
            self.metadata: pd.DataFrame = self.metadata[
                self.metadata.k_fold.isin(folds)
            ].reset_index(drop=True)
            self.image_path = self.data_root + train_image_path
        else:  # test images
            self.metadata: pd.DataFrame = pd.read_csv(data_root + test_csv_path)
            self.image_path = self.data_root + test_image_path

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
        # validation and test
        if (len(folds) == 1 and is_train) or not is_train:
            self.augmentations = albumentations.Compose(
                [
                    # albumentations.Resize(img_size, img_size, always_apply=True),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        # train
        else:
            self.augmentations = albumentations.Compose(
                [
                    # albumentations.Resize(img_size, img_size, always_apply=True),
                    albumentations.ShiftScaleRotate(rotate_limit=40, p=0.9),
                    albumentations.Normalize(always_apply=True),
                ]
            )

    def __len__(self):
        return len(self.metadata.image)

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        sample = {"image_id": data.image}

        if self.is_train:
            sample["individual_id"] = data.cat_id
            sample["species"] = data.species

        image = self.read_image(data.image)
        image = self.augmentations(image=image)["image"]
        image = torch.tensor(image, dtype=torch.float32)
        sample["image"] = image.permute(2, 0, 1)

        return sample

    @staticmethod
    def get_train_csv(path: str) -> pd.DataFrame:
        # fix spelling of two species and merging subspecies, reduces the unique species from 30 -> 26
        df: pd.DataFrame = pd.read_csv(path)
        df["species"].replace(
            {
                "bottlenose_dolpin": "bottlenose_dolphin",
                "kiler_whale": "killer_whale",
                "pilot_whale": "short_finned_pilot_whale",
                "globis": "short_finned_pilot_whale",
            },
            inplace=True,
        )
        return df

    def read_image(self, image_id: str) -> np.ndarray:
        image = io.imread(self.image_path + image_id)
        return image if len(image.shape) == 3 else gray2rgb(image)


if __name__ == "__main__":
    dataset = Whales(folds=[0, 1, 2, 3])
    print(dataset[1]["image"].shape)
    print(dataset[1]["individual_id"])
