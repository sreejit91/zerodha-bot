from setuptools import setup, find_packages

setup(
    name="algo",          # package name that ‘pip install’ will register
    version="0.1.0",
    packages=find_packages(),   # automatically includes your algo/ folder
)
