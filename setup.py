from setuptools import setup, find_packages

setup(
    name='MolOracle',
    version='0.1.0',
    author='Thomas Nevolianis',
    description='A package for predicting molecular properties using Graph Neural Networks',
    long_description=open('README.md').read(),
    url='https://github.com/thomasnevolianis/MolOracle',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
