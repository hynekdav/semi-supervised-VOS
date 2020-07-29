# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop
import click
import ffmpeg


@click.command(name='prediction-only')
@click.option('-p', '--prediction', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing predictions.')
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to save the output video.')
def prediction_only_command(prediction, output):
    pipeline = ffmpeg.input(prediction + '/*.png', pattern_type='glob', framerate=25)
    pipeline = pipeline.output(output)
    pipeline.run()
