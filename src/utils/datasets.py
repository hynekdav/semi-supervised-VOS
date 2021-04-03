# -*- encoding: utf-8 -*-
# ! python3
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from loguru import logger
from torchvision import datasets
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from .transforms import get_crop_params, crop, FixedColorJitter
from ..config import Config


class TrainDataset(datasets.ImageFolder):
    def __init__(self,
                 img_root,
                 annotation_root,
                 cropping=256,
                 frame_num=10,
                 transform=None,
                 target_transform=None,
                 color_jitter=False):
        super(TrainDataset, self).__init__(img_root,
                                           transform=transform,
                                           target_transform=target_transform)
        # img root and annotation root should have the same class_to_idx
        self.annotations = make_dataset(annotation_root, self.class_to_idx, extensions=('png', 'jpg', 'jpeg'))
        self.cropping = cropping
        self.frame_num = frame_num
        self.color_jitter = color_jitter
        self.rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        # read all jpgs and annotations into mem to speed up training
        self.img_bytes = []
        self.annotation_bytes = []

        logger.info(f'Loading {len(self.imgs)} train images.')
        for path, _ in tqdm(self.imgs):
            with Path(path).open(mode='rb') as f:
                self.img_bytes.append(f.read())
        logger.info(f"JPEGImages loaded: {len(self.img_bytes)}.")

        logger.info(f'Loading {len(self.imgs)} train annotations.')
        for path, _ in tqdm(self.annotations):
            with Path(path).open(mode='rb') as f:
                self.annotation_bytes.append(f.read())
        logger.info(f"Annotations loaded: {len(self.annotation_bytes)}.")

    def __getitem__(self, index):
        img_output = []
        annotation_output = []

        # if index reaches end of dataset, get the last frames
        if index + self.frame_num > len(self.imgs):
            index = len(self.imgs) - self.frame_num
        while not self.__is_from_same_video__(index):
            index -= 1
        # get transform params
        if self.color_jitter:
            color_transform = FixedColorJitter(brightness=0.4, contrast=0.4,
                                               saturation=0.4, hue=0.4)
        else:
            color_transform = lambda t: t
        crop_i, crop_j, th, tw = 0, 0, 0, 0
        h_flip = True if torch.rand(size=(1,)).item() < 0.5 else False
        v_flip = True if torch.rand(size=(1,)).item() < 0.5 else False
        for i in range(self.frame_num):
            img = Image.open(BytesIO(self.img_bytes[index + i]))
            img = img.convert('RGB')
            annotation = Image.open(BytesIO(self.annotation_bytes[index + i]))  # (W, H), -P mode
            annotation = annotation.convert('RGB')

            if h_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                annotation = annotation.transpose(Image.FLIP_LEFT_RIGHT)
            if v_flip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                annotation = annotation.transpose(Image.FLIP_TOP_BOTTOM)
            if i == 0:
                W, H = img.size
                crop_i, crop_j, th, tw = get_crop_params((W, H), self.cropping)

            # all images and annotations should cropped in the same way
            img_cropped = crop(img, crop_i, crop_j, th, tw)
            annotation_cropped = crop(annotation, crop_i, crop_j, th, tw)
            img_cropped = color_transform(img_cropped)

            img_cropped = self.rgb_normalize(img_cropped).numpy()
            annotation_cropped = np.asarray(annotation_cropped).transpose((2, 0, 1))
            img_output.append(img_cropped)
            annotation_output.append(annotation_cropped)

        img_output = torch.from_numpy(np.asarray(img_output)).float()
        annotation_output = torch.from_numpy(np.asarray(annotation_output)).float()
        _, video_index = self.imgs[index + self.frame_num - 1]
        return img_output, annotation_output, video_index

    def __is_from_same_video__(self, index):
        _, indexStart = self.imgs[index]
        _, indexEnd = self.imgs[index + self.frame_num - 1]
        return indexStart == indexEnd


class InferenceDataset(datasets.ImageFolder):
    """
        Load one frame at a time.
        Used for inference.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 disable=False,
                 inference_strategy='single'):
        super(InferenceDataset, self).__init__(root,
                                               transform=transform,
                                               target_transform=target_transform)
        self.img_bytes = []
        self.rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        logger.info(f'Loading {len(self.imgs)} inference images.')
        for path, _ in tqdm(self.imgs, disable=disable):
            with Path(path).open(mode='rb') as f:
                self.img_bytes.append(f.read())
        logger.info(f'Loaded {len(self.img_bytes)} inference images.')
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.inference_strategy = inference_strategy

    def __getitem__(self, index):
        path, video_index = self.imgs[index]
        img = Image.open(BytesIO(self.img_bytes[index]))
        img = img.convert('RGB')
        normalized = self.rgb_normalize(np.asarray(img))
        if self.inference_strategy == 'hor-flip':
            img = ImageOps.mirror(img)
            normalized_flipped = self.rgb_normalize(np.asarray(img))
            return (normalized, normalized_flipped), self.idx_to_class[video_index]
        elif self.inference_strategy == 'vert-flip':
            img = ImageOps.flip(img)
            normalized_flipped = self.rgb_normalize(np.asarray(img))
            return (normalized, normalized_flipped), self.idx_to_class[video_index]
        elif self.inference_strategy == '2-scale':
            img_2_size = np.ceil(np.array(img.size) / 2).astype(np.int)
            img_2 = img.resize(img_2_size, Image.ANTIALIAS)
            normalized_2 = self.rgb_normalize(np.asarray(img_2))
            return (normalized, normalized_2), self.idx_to_class[video_index]

        return normalized, self.idx_to_class[video_index]

    def __len__(self):
        return len(self.imgs)


class TripletLossTrainDataset(datasets.ImageFolder):
    def __init__(self,
                 img_root,
                 annotation_root,
                 transform=None,
                 target_transform=None):
        super(TripletLossTrainDataset, self).__init__(img_root,
                                                      transform=transform,
                                                      target_transform=target_transform)
        annotations_data = make_dataset(annotation_root, self.class_to_idx, extensions=('png', 'jpg', 'jpeg'))
        self.rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.annotation_convert = transforms.Compose([transforms.ToTensor()])
        self.data = defaultdict(lambda: [])

        assert len(self.imgs) == len(annotations_data)

        logger.info(f'Loading {len(self.imgs)} train image, annotation pairs.')
        for (image_path, image_sequence_idx), (annotation_path, annotation_sequence_idx) in tqdm(
                zip(self.imgs, annotations_data), total=len(self.imgs)):
            assert image_sequence_idx == annotation_sequence_idx
            with Path(image_path).open(mode='rb') as f:
                img = f.read()
            with Path(annotation_path).open(mode='rb') as f:
                annotation = f.read()

            self.data[image_sequence_idx].append((img, annotation))

        logger.info(f"Pairs loaded: {len(self.data)}.")

    def __getitem__(self, index):
        sequence = []

        for img, annotation in self.data[index]:
            img = Image.open(BytesIO(img))
            img = self.rgb_normalize(img.convert('RGB'))

            annotation = Image.open(BytesIO(annotation))
            annotation = annotation.convert('RGB')
            annotation = np.asarray(annotation).transpose((2, 0, 1))
            annotation = torch.from_numpy(annotation.copy()).float()

            sequence.append((img, annotation))

        return sequence

    def __len__(self):
        return len(self.data)
