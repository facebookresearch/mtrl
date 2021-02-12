# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Collection of utility functions"""

import os
import pathlib
import random
import re
import subprocess  # noqa: S404
from typing import Any, Iterator, List, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")


def flatten_list(_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list

    Args:
        _list (List[List[Any]]): List of lists

    Returns:
        List[Any]: Flattened list
    """
    return [item for sublist in _list for item in sublist]


def chunks(_list: List[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from given list.
    Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    Args:
        _list (List[T]): list to chunk.
        n (int): size of chunks.

    Yields:
        Iterator[List[T]]: iterable over the chunks
    """
    for index in range(0, len(_list), n):
        yield _list[index : index + n]  # noqa: E203


def make_dir(path: str) -> str:
    """Make a directory, along with parent directories.
    Does not return an error if the directory already exists.

    Args:
        path (str): path to make the directory.

    Returns:
        str: path of the new directory.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_current_commit_id() -> str:
    """Get current commit id.

    Returns:
        str: current commit id.
    """
    command = "git rev-parse HEAD"
    commit_id = (
        subprocess.check_output(command.split()).strip().decode("utf-8")  # noqa: S603
    )
    return commit_id


def has_uncommitted_changes() -> bool:
    """Check if there are uncommited changes.

    Returns:
        bool: wether there are uncommiteed changes.
    """
    command = "git status"
    output = subprocess.check_output(command.split()).strip().decode("utf-8")
    return "nothing to commit (working directory clean)" not in output


def set_seed(seed: int) -> None:
    """Set the seed for python, numpy, and torch.

    Args:
        seed (int): seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    # Module has no attribute "manual_seed_all"  [attr-defined]
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_on_caps(input_str: str) -> List[str]:
    """Split a given string at uppercase characters.
    Taken from: https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters

    Args:
        input_str (str): string to split.

    Returns:
        List[str]: splits of the given string.
    """
    return re.findall("[A-Z][^A-Z]*", input_str)


def is_integer(n: Union[int, str, float]) -> bool:
    """Check if the given value can be interpreted as an integer.

    Args:
        n (Union[int, str, float]): value to check.

    Returns:
        bool: can be the value be interpreted as an integer.
    """
    try:
        int(n)
    except ValueError:
        return False
    else:
        return True
