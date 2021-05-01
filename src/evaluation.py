# -*- encoding: utf-8 -*-
# ! python3
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from src.config import Config
from src.utils.metrics import evaluate_segmentation


def process_pair(gt, seg):
    gt_img = Image.open(gt).convert('P')
    seg_img = Image.open(seg).convert('P')
    seg_img = seg_img.resize(gt_img.size)

    gt_img = np.asarray(gt_img)
    seg_img = np.asarray(seg_img)

    gt_palette = np.unique(gt_img)
    seg_palette = np.unique(seg_img)

    scores = []
    for gt_color, seg_color in zip(gt_palette, seg_palette):
        gt_to_process = gt_img == gt_color
        seg_to_process = seg_img == seg_color
        score = evaluate_segmentation(gt_to_process, seg_to_process)
        scores.append(score)
    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)

    return mean_scores


@click.command(name='evaluation')
@click.option('--ground_truth', '-g', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to ground truth dataset folder')
@click.option('--computed_results', '-c', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to inference dataset folder')
def evaluation_command(ground_truth, computed_results):
    evaluation_command_impl(ground_truth, computed_results)


def evaluation_command_impl(ground_truth, computed_results, disable=False):
    computed = Path(computed_results)
    ground_truth = Path(ground_truth)

    ground_truth = list(ground_truth.glob('**/*.png'))
    computed = list(computed.glob('**/*.png'))

    ground_truth.sort()
    computed.sort()
    total = len(ground_truth)
    assert len(ground_truth) == len(computed)

    logger.info(f'Staring evaluation on {total} pairs.')

    pbar = tqdm(total=total, disable=disable)
    with Pool(Config.CPU_COUNT) as pool:
        res = [pool.apply_async(process_pair, args=(gt, seg,), callback=lambda _: pbar.update(1)) for gt, seg in
               zip(ground_truth, computed)]
        scores = [p.get() for p in res]
        pbar.close()
    scores = np.array(scores)
    j = scores[:, 0]
    f = scores[:, 1]
    j_mean = j.mean()
    f_mean = f.mean()
    jf_mean = np.array([j_mean, f_mean]).mean()
    logger.info(f'Evaluated: j_mean={j_mean}, f_mean={f_mean}, j&f_mean={jf_mean}.')
    return j_mean, f_mean, jf_mean
