# -*- encoding: utf-8 -*-
# ! python3


import click

from src.visualization.overlay import overlay_command
from src.visualization.prediction_only import prediction_only_command
from src.visualization.side_by_side import side_by_side_command


@click.group(name='cli')
def cli():
    pass


cli.add_command(overlay_command)
cli.add_command(side_by_side_command)
cli.add_command(prediction_only_command)

cli()
