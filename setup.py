#!usr/bin/env python3

import re
from os import path

from setuptools import find_packages, setup

package_name = "nobunaga"
root_dir = path.abspath(path.dirname(__file__))


def _requirements():
    return [
        name.rstrip()
        for name in open(path.join(root_dir, "requirements.txt"), encoding="utf-8").readlines()
    ]


with open(path.join(root_dir, package_name, "__init__.py"), encoding="utf-8") as f:
    init_text = f.read()
    version = re.search(r"__version__ = [\'\"](.+?)[\'\"]", init_text).group(1)
    author = re.search(r"__author__ =\s*[\'\"](.+?)[\'\"]", init_text).group(1)
    url = re.search(r"__url__ =\s*[\'\"](.+?)[\'\"]", init_text).group(1)

assert version
assert author
assert url

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="nobunaga",
    version=version,
    description="Nobunaga: Object Detection Analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    url=url,
    install_requires=_requirements(),
    include_package_data=True,
    license="",
    packages=find_packages(exclude=("tests")),
    test_suite="tests",
    entry_points="""
    [console_scripts]
        nobunaga = nobunaga.command_line:main
    """,
)
