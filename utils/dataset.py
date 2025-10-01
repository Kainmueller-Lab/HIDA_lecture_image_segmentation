import glob
import os
import torch
import zarr

import albumentations as A
import numpy as np

from torch.utils.data import Dataset

from utils.label import PredictionType



class KaggleDSB_dataset(Dataset):
    """(Subset of the) Kaggle Data Science Bowl 2018 dataset."""

    def __init__(
        self,
        root_dir,
        mode,
        prediction_type,
        padding_size=None
    ):
        self.mode = mode
        self.files = glob.glob(os.path.join(root_dir, mode, "*.zarr"))
        self.prediction_type = prediction_type
        self.padding_size = padding_size
        self.define_augmentation()
        self.define_padding()

    def __len__(self):
        return len(self.files)

    def define_augmentation(self):
        """Define the augmentation pipeline using Albumentations."""
        if self.mode == "train":
            augmentation_list = [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=45, border_mode=0, p=0.7),
            ]
            self.aug_transform = A.Compose(augmentation_list)
        else:
            self.aug_transform = A.Compose([])

    def define_padding(self):
        """Define the initial padding of the images."""
        if self.padding_size is not None:
            pad = self.padding_size
            def pad_image(image, **kwargs):
                return np.pad(
                    image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            self.pad_transform = A.Compose([
                A.Lambda(image=pad_image, mask=pad_image)
            ])
        else:
            self.pad_transform = A.Compose([])

    def get_filename(self, idx):
        """Get the filename of the idx-th sample."""
        return self.files[idx]

    def __getitem__(self, idx):
        fn = self.get_filename(idx)
        raw, label = self.load_sample(fn)
        raw = self.normalize(raw)

        # Padding
        if self.padding_size is not None:
            raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
            label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
            padded = self.pad_transform(image=raw, mask=label)
            raw = padded["image"]
            label = padded["mask"]
            raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
            label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

        # Augmentation (only during training)
        if self.mode == "train":
            raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
            label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
            raw, label = self.augment_sample(raw, label)
            raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
            label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

        raw, label = torch.tensor(raw.copy(), dtype=torch.float32), torch.tensor(label.copy(), dtype=torch.float32)
        return raw, label

    def augment_sample(self, raw, label):
        """Apply Albumentations augmentations."""
        augmented = self.aug_transform(image=raw, mask=label)
        raw_aug = augmented["image"]
        label_aug = augmented["mask"]
        return raw_aug, label_aug

    @staticmethod
    def normalize(raw):
        """Normalize the raw image to zero mean and unit variance."""
        raw -= np.mean(raw)
        raw /= np.std(raw)
        return raw

    def load_sample(self, filename):
        """Load a sample from a Zarr file."""
        data = zarr.open(filename)
        raw = np.array(data['volumes/raw'])

        if self.prediction_type == PredictionType.TWO_CLASS:
            label = np.array(data['volumes/gt_fgbg'])
        elif self.prediction_type == PredictionType.THREE_CLASS:
            label = np.array(data['volumes/gt_threeclass'])
        else:
            raise NotImplementedError

        label = label.astype(np.float32)
        return raw, label


class TestDataset(Dataset):
    """(Subset of the) Kaggle Data Science Bowl 2018 dataset."""

    def __init__(
        self,
        root_dir,
        mode,
        prediction_type,
        augmentation_list,
        padding_size=None
    ):
        self.mode = mode
        self.files = glob.glob(os.path.join(root_dir, mode, "*.zarr"))
        self.prediction_type = prediction_type
        self.augmentation_list = augmentation_list
        self.padding_size = padding_size
        self.define_augmentation()
        self.define_padding()

    def __len__(self):
        return len(self.files)

    def define_augmentation(self):
        """Define the augmentation pipeline using Albumentations."""
        if self.mode == "train":
            self.aug_transform = A.Compose(self.augmentation_list)
        else:
            self.aug_transform = A.Compose([])

    def define_padding(self):
        """Define the initial padding of the images."""
        if self.padding_size is not None:
            pad = self.padding_size
            def pad_image(image, **kwargs):
                return np.pad(
                    image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            self.pad_transform = A.Compose([
                A.Lambda(image=pad_image, mask=pad_image)
            ])
        else:
            self.pad_transform = A.Compose([])

    def get_filename(self, idx):
        """Get the filename of the idx-th sample."""
        return self.files[idx]

    def __getitem__(self, idx):
        fn = self.get_filename(idx)
        raw, label = self.load_sample(fn)
        raw = self.normalize(raw)

        # Padding
        if self.padding_size is not None:
            raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
            label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
            padded = self.pad_transform(image=raw, mask=label)
            raw = padded["image"]
            label = padded["mask"]
            raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
            label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

        # Augmentation (only during training)
        if self.mode == "train":
            raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
            label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
            raw, label = self.augment_sample(raw, label)
            raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
            label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

        raw, label = torch.tensor(raw.copy(), dtype=torch.float32), torch.tensor(label.copy(), dtype=torch.float32)
        return raw, label

    def augment_sample(self, raw, label):
        """Apply Albumentations augmentations."""
        augmented = self.aug_transform(image=raw, mask=label)
        raw_aug = augmented["image"]
        label_aug = augmented["mask"]
        return raw_aug, label_aug

    @staticmethod
    def normalize(raw):
        """Normalize the raw image to zero mean and unit variance."""
        raw -= np.mean(raw)
        raw /= np.std(raw)
        return raw

    def load_sample(self, filename):
        """Load a sample from a Zarr file."""
        data = zarr.open(filename)
        raw = np.array(data['volumes/raw'])

        if self.prediction_type == PredictionType.TWO_CLASS:
            label = np.array(data['volumes/gt_fgbg'])
        elif self.prediction_type == PredictionType.THREE_CLASS:
            label = np.array(data['volumes/gt_threeclass'])
        else:
            raise NotImplementedError

        label = label.astype(np.float32)
        return raw, label
