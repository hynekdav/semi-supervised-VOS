# -*- encoding: utf-8 -*-
# ! python3

from pathlib import Path

import click
import numpy as np
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from multiprocessing import Pool

from src.config import Config
from src.utils.metrics import evaluate_segmentation


def process_pair(gt, seg):
    gt_img = np.ceil(imread(str(gt), as_gray=True))
    seg_img = np.ceil(imread(str(seg), as_gray=True))
    return evaluate_segmentation(gt_img, seg_img)


@click.command(name='evaluation')
@click.option('--ground_truth', '-g', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to ground truth dataset folder')
@click.option('--computed_results', '-c', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to inference dataset folder')
def evaluation_command(ground_truth, computed_results):
    computed = Path(computed_results)
    ground_truth = Path(ground_truth)

    ground_truth = list(ground_truth.glob('**/*.png'))
    computed = list(computed.glob('**/*.png'))

    ground_truth.sort()
    computed.sort()
    total = len(ground_truth)
    assert len(ground_truth) == len(computed)

    logger.info(f'Staring evaluation on {total} pairs.')
    pbar = tqdm(total=total)
    with Pool(Config.CPU_COUNT) as pool:
        res = [pool.apply_async(process_pair, args=(gt, seg,), callback=lambda _: pbar.update(1)) for gt, seg in
               list(zip(ground_truth, computed))]
        scores = [p.get() for p in res]
        pbar.close()
    j, f = map(np.array, zip(*scores))
    logger.info(f'Evaluated: j_mean={j.mean()}, f_mean={f.mean()}.')
