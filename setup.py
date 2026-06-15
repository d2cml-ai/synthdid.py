from pathlib import Path

from setuptools import find_packages, setup


long_description = Path(__file__).with_name("Readme.md").read_text(encoding="utf-8")

setup(
    dependency_links=[],
    install_requires=[
        "matplotlib>=3.7",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "statsmodels>=0.14",
    ],
    name="synthdid",
    author="D2CML Team, Alexander Quispe, Rodrigo  Grijalba, Jhon Flores, Franco Caceres",
    version="0.10.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="causal-inference",
    url="https://github.com/d2cml-ai/synthdid.py",
    license="MIT",
    description="Synthdid",
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
    ],
)
