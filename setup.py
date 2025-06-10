from setuptools import setup, find_packages

setup(
    name='openbiomed',
    version='0.1.0',
    packages=find_packages(include=["openbiomed", "openbiomed.*"]),
    install_requires=[],
)