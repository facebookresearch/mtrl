# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
import codecs
import os.path

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def parse_dependency(filepath):
    dep_list = []
    for dep in open(filepath).read().splitlines():
        if dep.startswith("#"):
            continue
        key = "#egg="
        if key in dep:
            git_link, egg_name = dep.split(key)
            dep = f"{egg_name} @ {git_link}"
        dep_list.append(dep)
    return dep_list


base_requirements = parse_dependency("requirements/base.txt")
dev_requirements = base_requirements + parse_dependency("requirements/dev.txt")


extras_require = {}

extras_require["dev"] = dev_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtrl",
    version=get_version("mtrl/__init__.py"),
    author="Shagun Sodhani, Amy Zhang",
    author_email="sshagunsodhani@gmail.com, amyzhang@fb.com",
    description="MTRL: Multi Task RL Algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=base_requirements,
    url="https://github.com/facbookresearch/mtrl",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs", "docsrc"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    extras_require=extras_require,
)
