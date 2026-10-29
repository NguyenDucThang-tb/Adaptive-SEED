"""State shared explicitly between research stages, isolated for each run."""
from dataclasses import dataclass
from argparse import Namespace


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


@dataclass
class Runtime:
    args: Namespace
    display = staticmethod(display)
