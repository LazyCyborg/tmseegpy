# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tmseegpy",
    version="0.1.0",
    author="LazyCyborg",
    author_email="hjarneko@gmail.com",
    description="A pipeline for preprocessing and analyzing TMS-EEG data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LazyCyborg/tmseegpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy >= 1.16.0',
        'scipy >= 1.2.0',
        'mne >= 1.1',
        'pooch',
        'torch',
        "scikit-learn",
        "tensorly",
        "matplotlib",
        "seaborn",
        "tqdm",
        "mne-faster",
        'psutil',
        'construct',
        'importlib-metadata; python_version<"3.8"',
        'importlib-resources; python_version<"3.9"',

    ],
    entry_points={
        "console_scripts": [
            "tmseegpy=tmseegpy.gui:main",
        ],
    },
)