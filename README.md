[![CircleCI](https://circleci.com/gh/facebookresearch/mtrl.svg?style=svg&circle-token=8cc8eb1b9666a65e27a21c39b5d5398744365894)](https://circleci.com/gh/facebookresearch/mtrl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# MTRL
Multi Task RL Algorithms

## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Usage](#Usage)

4. [Documentation](#Documentation)

5. [Contributing to MTRL](#Contributing-to-MTRL)

6. [Community](#Community)

7. [Acknowledgements](#Acknowledgements)

## Introduction

MTRL is a library of multi-task reinforcement learning algorithms. It has two main components:

* [Building blocks](https://github.com/facebookresearch/mtrl/tree/main/mtrl/agent/components) and [agents](https://github.com/facebookresearch/mtrl/tree/main/mtrl/agent) that implement the multi-task RL algorithms.

* [Experiment setups](https://github.com/facebookresearch/mtrl/tree/main/mtrl/experiment) that enable training/evaluation on different setups. 

Together, these two components enable use of MTRL across different environments and setups.

### List of publications & submissions using MTRL (please create a pull request to add the missing entries):

* [Learning Robust State Abstractions for Hidden-Parameter Block MDPs](https://arxiv.org/abs/2007.07206)
* [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/abs/2102.06177)
    *  We use the `af8417bfc82a3e249b4b02156518d775f29eb289` commit for the MetaWorld environments for our experiments.

### License

* MTRL uses [MIT License](https://github.com/facebookresearch/mtrl/blob/main/LICENSE).

* [Terms of Use](https://opensource.facebook.com/legal/terms)

* [Privacy Policy](https://opensource.facebook.com/legal/privacy)

### Citing MTRL

If you use MTRL in your research, please use the following BibTeX entry:
```
@Misc{Sodhani2021MTRL,
  author =       {Shagun Sodhani and Amy Zhang},
  title =        {MTRL - Multi Task RL Algorithms},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/facebookresearch/mtrl}
}
```

## Setup

* Clone the repository: `git clone git@github.com:facebookresearch/mtrl.git`.

* Install dependencies: `pip install -r requirements/dev.txt`

## Usage

* MTRL supports 8 different multi-task RL algorithms as described [here](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html).

* MTRL supports multi-task environments using [MTEnv](https://github.com/facebookresearch/mtenv). These environments include [MetaWorld](https://meta-world.github.io/) and multi-task variants of [DMControl Suite](https://github.com/deepmind/dm_control)

* Refer the [tutorial](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html) to get started with MTRL.

## Documentation

[https://mtrl.readthedocs.io](https://mtrl.readthedocs.io)

## Contributing to MTRL

There are several ways to contribute to MTRL.

1. Use MTRL in your research.

2. Contribute a new algorithm. We currently support [8 multi-task RL algorithms](https://mtrl.readthedocs.io/en/latest/pages/algorithms/supported.html) and are looking forward to adding more environments.

3. Check out the [good-first-issues](https://github.com/facebookresearch/mtrl/pulls?q=is%3Apr+is%3Aopen+label%3A%22good+first+issue%22) on GitHub and contribute to fixing those issues.

4. Check out additional details [here](https://github.com/facebookresearch/mtrl/blob/main/.github/CONTRIBUTING.md).

## Community

Ask questions in the chat or github issues:
* [Chat](https://mtenv.zulipchat.com)
* [Issues](https://github.com/facebookresearch/mtrl/issues)

## Acknowledgements

* Our implementation of SAC is inspired by Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).
