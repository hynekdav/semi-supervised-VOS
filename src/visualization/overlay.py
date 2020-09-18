# -*- encoding: utf-8 -*-
# ! python3



from pathlib import Path

import click
import ffmpeg
import numpy as np
from PIL import Image


def remove_background(path):
    files = Path(path).glob('*.png')
    for file in files:
        image = Image.open(file)
        image = image.convert('RGBA')
        data = np.array(image)
        rgb = data[:, :, :3]
        color = [0, 0, 0]
        mask = np.all(rgb == color, axis=-1)
        data[mask] = [0, 0, 0, 0]
        data[np.logical_not(mask), 3] = 128

        new_im = Image.fromarray(data)
        new_im.save(str(file.absolute()) + '.noback.png')


def cleanup(path):
    files = Path(path).glob('*.noback.png')
    for file in files:
        file.unlink(missing_ok=True)


@click.command(name='overlay')
@click.option('-p', '--prediction', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing predictions.')
@click.option('-s', '--source', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing video frames.')
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to save the output video.')
def overlay_command(prediction, source, output):
    remove_background(prediction)
    prediction_input = ffmpeg.input(prediction + '/*.noback.png', pattern_type='glob', framerate=25, vcodec='png')
    source_input = ffmpeg.input(source + '/*.jpg', pattern_type='glob', framerate=25)

    source_input = source_input.overlay(prediction_input)
    pipeline = source_input.output(output)
    pipeline.run()
    cleanup(prediction)
