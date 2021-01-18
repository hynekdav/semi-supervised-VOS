# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import torch
from loguru import logger
from tqdm import tqdm

from src.evaluation import evaluation_command_impl
from src.inference import inference_command_impl


def process_model(data, model_path):
    with TemporaryDirectory(prefix=model_path.stem) as out_dir:
        inference_command_impl(10, data, model_path, 'resnet50',
                               1.0, 40, 8.0, 8.0, out_dir,
                               torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), True)
        j_mean, f_mean = evaluation_command_impl(data / 'Annotations/480p', out_dir, True)
    loss = str(model_path.stem).split('-')[-1]
    return loss, j_mean, f_mean


@click.command(name='prepare_charting_data')
@click.option('--data', type=click.Path(exists=True, file_okay=False), help='Path to dataset.', required=True)
@click.option('--models', type=click.Path(exists=True, file_okay=False), help='Path to models.', required=True)
@click.option('--output', type=click.Path(exists=False, dir_okay=False), help='Path to save output to.',
              default='output.json')
def prepare_charting_data_command(data, models, output):
    original_level = os.environ.get('LOGURU_LEVEL')
    logger.remove()
    logger.add(sys.stderr, level='ERROR')
    models = Path(models)
    data = Path(data)
    output = Path(output)

    models = list(models.glob('**/*.pth.tar'))
    models.sort()
    for model_path in tqdm(models, desc='Evaluating all models.'):
        logger.remove()
        logger.add(sys.stderr, level='ERROR')

        loss_type = model_path.stem.replace('.pth', '').replace('.tar', '')
        loss, j_mean, f_mean = process_model(data, model_path)

        if output.exists():
            with output.open(mode='r') as out:
                results = json.load(out)
        else:
            results = {}

        if loss_type not in results:
            results[loss_type] = {'loss': [], 'j_mean': [], 'f_mean': []}
        results[loss_type]['loss'].append(loss)
        results[loss_type]['j_mean'].append(j_mean)
        results[loss_type]['f_mean'].append(f_mean)

        logger.remove()
        logger.add(sys.stderr, level=original_level or 'DEBUG')

        with output.open(mode='w') as out:
            json.dump(results, out, indent=4)
