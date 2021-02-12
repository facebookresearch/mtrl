# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""This is the main entry point for the running the experiments."""

import hydra
from ml_logger.logbook import LogBook

from mtrl.experiment import utils as experiment_utils
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType


def run(config: ConfigType) -> None:
    """Create and run the experiment.

    Args:
        config (ConfigType): config for the experiment.
    """
    config_utils.pretty_print(config, resolve=False)
    config_id = config.setup.id
    logbook_config = hydra.utils.call(config.logbook)
    if "mongo" in logbook_config["loggers"] and (
        config_id.startswith("pytest_")
        or config_id in ["sample", "sample_config"]
        or config_id.startswith("test_")
        # or is_debug_job
    ):
        # do not write the job to mongo db.
        print(logbook_config["loggers"].pop("mongo"))
    logbook = LogBook(logbook_config)
    config_to_write = config_utils.to_dict(config)

    config_to_write["status"] = "RUNNING"
    logbook.write_metadata(config_to_write)

    experiment_utils.prepare_and_run(config=config)

    config_to_write["status"] = "COMPLETED"
    logbook.write_metadata(config_to_write)
