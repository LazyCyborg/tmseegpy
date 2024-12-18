# setup.py
from setuptools import setup, find_packages

setup(
    name="tmseegpy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "mne",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'tmseegpy=tmseegpy.main:main',
        ],
    },
    author="LazyCyborg",
    author_email="your.email@example.com",
    description="A TMS-EEG analysis package with GUI interface",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LazyCyborg/tmseegpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
