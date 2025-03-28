from setuptools import setup, find_packages

setup(
    name="adalasso",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.16.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.0.0",
    ],
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    description="Adaptive Lasso implementation following scikit-learn API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ErikHartman/adalasso",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
)
