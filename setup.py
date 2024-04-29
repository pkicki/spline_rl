from setuptools import setup, find_packages
from spline_rl import __version__



setup(author="Piotr Kicki",
      url="https://github.com/robfiras/loco-mujoco",
      version=__version__,
      packages=[package for package in find_packages()
                if package.startswith('spline_rl')],
      )
