from setuptools import setup, find_packages

setup(
    name="longrange",
    version="0.1",
    packages=find_packages({"lr", "lrgb", "helpers", "longrange"}),
    package_dir={"": "."},
)
