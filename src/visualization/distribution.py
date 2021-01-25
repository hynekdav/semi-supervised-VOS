# -*- encoding: utf-8 -*-
# ! python3
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from src.config import Config
from src.model.vos_net import VOSNet
from src.utils.utils import load_model, color_to_class


def remove_black_alpha(image):
    data = image.getdata()
    new_data = []

    for pixel in data:
        if pixel == (0, 0, 0, 255):
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(pixel)
    image.putdata(new_data)
    return image


def set_device_and_load_model(checkpoint, device):
    if Config.DEVICE.type != device:
        Config.DEVICE = torch.device(device)
    if not torch.cuda.is_available():
        Config.DEVICE = torch.device('cpu')
    model = VOSNet()
    try:
        model = load_model(model, checkpoint)
    except Exception:
        model = torch.nn.DataParallel(model)
        model = load_model(model, checkpoint)
    model = model.to(Config.DEVICE)
    model.eval()
    return model


def get_similarity_vector(features):
    features_norm = torch.nn.functional.normalize(features, p=1, dim=1)
    similarity = 1 - torch.cdist(features_norm, features_norm, p=1).squeeze().detach().numpy()
    indices = np.triu_indices(similarity.shape[0], k=1)
    similarity = similarity[indices].flatten()
    return similarity

def filter_labels(labels: torch.tensor):
    labels = labels.numpy()
    diff = np.diff(labels)
    idx = np.where(diff != 1)[0][0] + 1
    labels = labels[:idx]
    return labels

@click.command(name='distribution')
@click.option('-i', '--image', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to original image.')
@click.option('-a', '--annotation', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to image annotation.')
@click.option('-c', '--checkpoint', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to model checkpoint.')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda', help='Device to run computing on.')
@click.option('--save/--no-save', default=False, help='Save the image or show it.')
@click.option('-s', '--save_path', help='Path to save image to.')
def distribution_command(image, annotation, checkpoint, device, save, save_path):
    distribution_command_impl(image, annotation, checkpoint, device, save, save_path)


def distribution_command_impl(image, annotation, checkpoint, device, save, save_path):
    image_path = Path(image)
    model = set_device_and_load_model(checkpoint, device)
    annotation = Image.open(annotation).convert('RGB')
    annotation = torch.from_numpy(np.asarray(annotation)).permute((2, 0, 1)).unsqueeze(0).float()
    annotation_input_downsample = torch.nn.functional.interpolate(annotation,
                                                                  scale_factor=Config.SCALE,
                                                                  mode='bilinear',
                                                                  align_corners=False).to(Config.DEVICE)
    centroids = np.load("./annotation_centroids.npy")
    centroids = torch.Tensor(centroids).float().to(Config.DEVICE)
    annotation = color_to_class(annotation_input_downsample, centroids).squeeze().reshape(-1).cpu()

    rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    image = Image.open(image).convert('RGB')
    image_normalized = rgb_normalize(np.asarray(image)).unsqueeze(0).to(Config.DEVICE)
    features_tensor: torch.Tensor = model(image_normalized).detach().cpu().squeeze().permute((1, 2, 0)).reshape(-1, 256)
    features_tensor = features_tensor[:annotation.shape[0]]

    unique_labels = filter_labels(torch.unique(annotation))

    similarities = []
    for label in unique_labels:
        labels = (annotation == label).nonzero(as_tuple=False).squeeze()
        similarity = get_similarity_vector(features_tensor.index_select(0, labels))
        similarities.append(similarity)

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    fig, ax = plt.subplots(nrows=1, ncols=len(similarities), figsize=(len(similarities) * 8.1, 4.5))
    fig.suptitle(f'Similarity distributions - {image_path.parent.stem} frame: {image_path.stem.title()}')
    for idx, (similarity, color) in enumerate(zip(similarities, colors)):
        ax[idx].hist(similarity, bins=100, density=False, color=color)
        ax[idx].title.set_text(f'Label: {idx}')

    if save:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(save_path.absolute() / f'{image_path.parent.stem}_{image_path.stem.title()}-distribution.jpg'))
    else:
        fig.show()
    plt.close(fig)
