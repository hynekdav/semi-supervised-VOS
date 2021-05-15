# -*- encoding: utf-8 -*-
# ! python3


import click
import ffmpeg


@click.command(name='side-by-side')
@click.option('-p', '--prediction', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing predictions.')
@click.option('-s', '--source', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Directory containing video frames.')
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='Path to save the output video.')
@click.option('-f', '--fps', type=click.INT, default=15, required=False,
              help='Number of frames per second in the resulting video.')
def side_by_side_command(prediction, source, output, fps):
    prediction_input = ffmpeg.input(prediction + '/*.png', pattern_type='glob', framerate=fps, vcodec='png')
    source_input = ffmpeg.input(source + '/*.jpg', pattern_type='glob', framerate=fps)
    joined = ffmpeg.filter((prediction_input, source_input), 'hstack')
    pipeline = joined.output(output)
    pipeline.run()
