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
def side_by_side_command(prediction, source, output):
    prediction_input = ffmpeg.input(prediction + '/*.png', pattern_type='glob', framerate=25, vcodec='png')
    source_input = ffmpeg.input(source + '/*.jpg', pattern_type='glob', framerate=25)
    joined = ffmpeg.filter((prediction_input, source_input), 'hstack')
    pipeline = joined.output(output)
    pipeline.run()
