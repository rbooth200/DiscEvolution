#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

version = "1.0.0"

setup(
    name="DiscEvolution",
    version=version,
    packages=find_packages(),
    author="Richard Booth",
    author_email="richardabooth@gmail.com",
    description="A dust-gas evolution code for protoplanetary discs",
    long_description=open('README.md').read(),
    install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
    license="GPLv3",
    url="hhttps://github.com/rbooth200/DiscEvolution/",
    package_data={'':['data/*/*.npy', 'data/*/*.txt', 'FRIED/friedgrid.dat']},
    include_package_data=True,
    classifiers=[
         "Development Status :: 1 - Production/Stable",
         "Intended Audience :: Developers",
         "Intended Audience :: Science/Research",
         "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
         "Programming Language :: Python :: 3",
    ]
)
