# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fishswim',
    version='0.0.1',
    description='A fishy genetic algorithm.',
    long_description=readme,
    author='Ryan Brigden',
    author_email='rbrigden@cmu.edu',
    url='https://github.com/rbrigden/fishswim',
    license=license,
    packages=find_packages(exclude=('data', 'log'))
)
