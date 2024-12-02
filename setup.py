from setuptools import setup, find_packages

setup(
    name='tmseeg_preprocessing',
    version='0.1.0',
    description='A Python package for preprocessing TMS-EEG data using MNE',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/tmseeg_preprocessing',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'mne',
        'mne_icalabel',
        'mne_faster',
        'scipy',
        'seaborn',
        'tqdm',
        'pandas',
        'scikit-learn',
        'joblib',
        'statsmodels',
        'pyyaml',
        'pytest',
        'sklearn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.7',
)
