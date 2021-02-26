# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
import json
import os
import time
from functools import singledispatch
from typing import Dict, List

import numpy as np
import torch
from termcolor import colored


@singledispatch
def serialize_log(val):
    """Used by default."""
    return val


@serialize_log.register(np.float32)
def np_float32(val):
    return np.float64(val)


@serialize_log.register(np.int64)
def np_int64(val):
    return int(val)


class Meter(object):
    def __init__(self):
        pass

    def update(self, value, n=1):
        pass

    def value(self):
        pass


class AverageMeter(Meter):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class CurrentMeter(Meter):
    def __init__(self):
        pass

    def update(self, value, n=1):
        self._value = value

    def value(self):
        return self._value


class MetersGroup(object):
    def __init__(self, file_name, formating, mode: str, retain_logs: bool):
        self._file_name = file_name
        self._mode = mode
        if not retain_logs:
            if os.path.exists(file_name):
                os.remove(file_name)
        self._formating = formating
        self._meters: Dict[str, Meter] = {}

    def log(self, key, value, n=1):
        if key not in self._meters:
            metric_type = self._formating[key][2]
            if metric_type == "average":
                self._meters[key] = AverageMeter()
            elif metric_type == "constant":
                self._meters[key] = CurrentMeter()
            else:
                raise ValueError(f"{metric_type} is not supported by logger.")
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            data[key] = meter.value()
        data["mode"] = self._mode
        return data

    def _dump_to_file(self, data):
        data["logbook_timestamp"] = time.strftime("%I:%M:%S%p %Z %b %d, %Y")
        with open(self._file_name, "a") as f:
            f.write(json.dumps(data, default=serialize_log) + "\n")

    def _format(self, key, value, ty):
        template = "%s: "
        if ty == "int":
            template += "%d"
        elif ty == "float":
            template += "%.04f"
        elif ty == "time":
            template += "%.01f s"
        elif ty == "str":
            template += "%s"
        else:
            raise "invalid format type: %s" % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = ["{:5}".format(prefix)]
        for key, (disp_key, ty, _) in self._formating.items():
            if key in data:
                value = data.get(key, 0)
                if disp_key is not None:
                    pieces.append(self._format(disp_key, value, ty))
        print("| %s" % (" | ".join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["step"] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, config, retain_logs: bool = False):
        self._log_dir = log_dir
        self.config = config

        if "metaworld" in self.config.env.name:
            num_envs = int(
                "".join(
                    [
                        x
                        for x in self.config.env.benchmark._target_.split(".")[1]
                        if x.isdigit()
                    ]
                )
            )
        else:
            env_list: List[str] = []
            for key in self.config.metrics:
                if "_" in key:
                    mode, submode = key.split("_")
                    # todo: should we instead throw an error here?
                    if mode in self.config.env and submode in self.config.env[mode]:
                        env_list += self.config.env[mode][submode]
                else:
                    if key in self.config.env:
                        env_list += self.config.env[key]
            num_envs = len(set(env_list))

        def _get_formatting(
            current_formatting: List[List[str]],
        ) -> Dict[str, List[str]]:
            formating: Dict[str, List[str]] = {
                _format[0]: _format[1:] for _format in current_formatting
            }
            if num_envs > 0:
                keys = list(formating.keys())
                for key in keys:
                    if key.endswith("_"):
                        value = formating.pop(key)
                        for index in range(num_envs):
                            new_key = key + str(index)
                            if value[0] is None:
                                abbr = None
                            else:
                                abbr = value[0] + str(index)
                            formating[new_key] = [abbr, *value[1:]]
            return formating

        self.mgs = {
            key: MetersGroup(
                os.path.join(log_dir, f"{key}.log"),
                formating=_get_formatting(current_formatting=value),
                mode=key,
                retain_logs=retain_logs,
            )
            for key, value in self.config.metrics.items()
        }

    def log(self, key, value, step, n=1):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        mode, key = key.split("/", 1)
        self.mgs[mode].log(key, value, n)

    def dump(self, step):
        for key in self.mgs:
            self.mgs[key].dump(step, key)
