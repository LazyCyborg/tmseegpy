from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tmseegpy",
    version="0.1.8",
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
    python_requires=">=3.9,<3.10",
    install_requires=[
        # Core Processing Dependencies
        'numpy==1.25.2',
        'scipy==1.11.2',
        'mne>=1.1',
        'scikit-learn==1.3.0',
        'tensorly',
        'pandas==2.1.0',

        # GUI and Visualization
        'PyQt6>=6.4.0',
        'matplotlib==3.9.2',
        'seaborn',

        # Server Dependencies
        'Flask',
        'Flask-Cors',
        'Flask-SocketIO',
        'eventlet',
        'appdirs',

        # File Handling
        'pooch',
        'construct',

        # Progress and System
        'tqdm',
        'psutil',
        'mne-faster==1.2',

        # Additional Processing
        'edfio',
        'eeglabio',
        'antropy==0.1.6',
        'mne-icalabel',

        # Development and Testing
        'pytest',
        'pylint',
        'black'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pylint',
            'black',
            'sphinx',
            'sphinx-rtd-theme'
        ]
    },
    entry_points={
        'console_scripts': [
            'tmseegpy=tmseegpy:run_cli',
            'tmseegpy-server=tmseegpy:run_server_cmd',
        ],
    },
    package_data={
        'tmseegpy': ['config/*.json', 'data/*.fif', 'data/*.csv'],
    },
    include_package_data=True
)