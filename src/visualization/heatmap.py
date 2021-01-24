# -*- encoding: utf-8 -*-
# ! python3


import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import rescale
from torchvision.transforms import transforms

from src.config import Config
from src.model.vos_net import VOSNet
from src.utils.utils import load_model


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


def get_similarity_matrix(features):
    (_, _, H, W) = features.shape
    features = features.reshape(256, -1).permute(1, 0)
    features_norm = torch.nn.functional.normalize(features, p=1, dim=1)
    idx = torch.randint(low=500, high=features.shape[0] - 500, size=(1,))
    similarity = 1 - torch.cdist(features_norm[idx].reshape(1, -1), features_norm, p=1).squeeze().detach()
    similarity[idx] = 2
    similarity = similarity.reshape(H, W)
    return similarity, idx


@click.command(name='heatmap')
@click.option('-i', '--image', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to original image.')
@click.option('-a', '--annotation', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to image annotation.')
@click.option('-c', '--checkpoint', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to model checkpoint.')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda', help='Device to run computing on.')
def heatmap_command(image, annotation, checkpoint, device):
    model = set_device_and_load_model(checkpoint, device)
    annotation = remove_black_alpha(Image.open(annotation).convert('RGBA'))

    rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    image = Image.open(image).convert('RGB')
    image_normalized = rgb_normalize(np.asarray(image)).unsqueeze(0)
    features_tensor: torch.Tensor = model(image_normalized).detach().cpu()
    similarity, idx = get_similarity_matrix(features_tensor)
    x, y = int((idx % features_tensor.shape[3]) * 8), int((idx / features_tensor.shape[3]) * 8)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 8.1, 4.5))
    fig.suptitle(f'Selected index [{x}, {y}]')
    ax[0].imshow(annotation)
    ax[0].plot(x, y, 'bx')
    ax[0].title.set_text('Annotation mask')
    ax[1].imshow(image)
    ax[1].plot(x, y, 'bx')
    ax[1].title.set_text('Original image')
    ax[2].imshow(annotation)
    ax[2].imshow(rescale(similarity * 255, 8), alpha=0.8)
    ax[2].title.set_text('Heatmap')
    fig.show()
