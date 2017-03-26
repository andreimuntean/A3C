"""Installs the modules required to run the package."""

from setuptools import setup


setup(
    name='Asynchronous Advantage Actor-Critic',
    version='1.0.0',
    url='https://github.com/andreimuntean/a3c',
    description='Deep reinforcement learning using an asynchronous advantage actor-critic model.',
    author='Andrei Muntean',
    keywords='deep learning machine reinforcement neural network a3c actor-critic openai',
    install_requires=['gym[atari]', 'numpy', 'pillow', 'scipy', 'tensorflow']
)
