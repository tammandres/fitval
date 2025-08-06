from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
    name="fitval",
    version="0.0.1",
    author="Andres Tamm",
    description="External validation of prediction models involving the FIT test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tammandres/fitval",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={"console_scripts": ["fitval = bin.cli:main"]},
)