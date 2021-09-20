from distutils.core import setup
from platform import platform

from setuptools import find_packages




setup(
    name='rl',
    version=1.0,
    install_requires=[
        'cloudpickle',
#        'gym[atari,box2d,classic_control]',
#   	 'pybullet',
#        'matplotlib',
#        'numpy',
#        'torch',
    ],
    description="RL-AMMI tools for combining deep RL algorithms.",
    authors="MohamedElfatih Salah, Rami Ahmed, Ruba Mutasim, Wafaa Mohammed",
    url="https://github.com/RamiSketcher/RL-AMMI",
    author_email="rahmed@aimsammi.com"
)
