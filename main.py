# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import click

from src.inference import inference_command
from src.train import train_command


@click.group(name='cli')
def cli():
    pass

cli.add_command(inference_command)
cli.add_command(train_command)

cli()
