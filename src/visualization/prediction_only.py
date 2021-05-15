# -*- encoding: utf-8 -*-
# ! python3


import click
import ffmpeg


@click.command(name='prediction-only')
@click.option('-p', '--prediction', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing predictions.')
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to save the output video.')
@click.option('-f', '--fps', type=click.INT, default=15, required=False,
              help='Number of frames per second in the resulting video.')
def prediction_only_command(prediction, output, fps):
    pipeline = ffmpeg.input(prediction + '/*.png', pattern_type='glob', framerate=fps)
    pipeline = pipeline.output(output)
    pipeline.run()
