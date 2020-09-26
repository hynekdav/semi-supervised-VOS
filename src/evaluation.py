# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

from pathlib import Path

import click
import numpy as np
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from src.utils.metrics import evaluate_segmentation


@click.command(name='evaluation')
@click.option('--ground_truth', '-g', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to ground truth dataset folder')
@click.option('--computed_results', '-c', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to inference dataset folder')
def evaluation_command(ground_truth, computed_results):
    computed = Path(computed_results)
    ground_truth = Path(ground_truth)

    total = len(list(ground_truth.glob('**/*.png')))
    logger.info(f'Staring evaluation on {total} pairs.')
    scores = []
    for gt, seg in tqdm(zip(ground_truth.glob('**/*.png'), computed.glob('**/*.png')), total=total):
        gt_img = np.ceil(imread(str(gt), as_gray=True))
        seg_img = np.ceil(imread(str(seg), as_gray=True))
        scores.append(evaluate_segmentation(gt_img, seg_img))
    j, f = map(np.array, zip(*scores))
    logger.info(f'Evaluated: j_mean={j.mean()}, f_mean={f.mean()}.')
