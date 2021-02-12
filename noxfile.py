# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
import os

import nox
from nox.sessions import Session

DEFAULT_PYTHON_VERSIONS = ["3.7", "3.8", "3.9"]

PYTHON_VERSIONS = os.environ.get(
    "NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)
).split(",")

code_paths_to_test = ["mtrl", "tests", "main.py"]
config_paths_to_test = ["config"]


def setup(session: Session) -> None:
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements/nox.txt")


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    setup(session)
    for _path in code_paths_to_test:
        session.run("flake8", _path)
        session.run("black", "--check", _path)
        session.run("isort", _path, "--check", "--diff")


@nox.session(python=PYTHON_VERSIONS)
def mypy(session):
    setup(session)
    for _path in code_paths_to_test:
        session.run("mypy", _path)


@nox.session(python=PYTHON_VERSIONS)
def yamllint(session):
    setup(session)
    for _path in config_paths_to_test:
        session.run("yamllint", _path)


# @nox.session(python=PYTHON_VERSIONS)
# # @nox.session(venv_backend="conda")
# def pytest(session):
#     session.install("--upgrade", "setuptools", "pip")
#     session.install("-r", "requirements.txt")
#     session.run("pytest", "tests", env={"PYTHONPATH": "."})
