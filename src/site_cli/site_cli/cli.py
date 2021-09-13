import click

from .active_learning import active_learning
from .inference import inference
from .retrain import retrain


@click.group()
def entry_point():
    """
    stcrayon is a CLI of the SITE AI automation project.
    """
    pass


entry_point.add_command(inference.inference)
entry_point.add_command(active_learning.activelearning)
entry_point.add_command(retrain.retrain)
