# -*- encoding: utf-8 -*-
# ! python3


import click

from src.inference import inference_command
from src.train import train_command, validation_command
from src.evaluation import evaluation_command


@click.group(name='cli')
def cli():
    pass


cli.add_command(inference_command)
cli.add_command(train_command)
cli.add_command(validation_command)
cli.add_command(evaluation_command)

cli()
