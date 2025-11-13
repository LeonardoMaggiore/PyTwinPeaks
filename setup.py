from setuptools import setup, find_packages

setup(
    name='pyTwinPeaks',
    version='0.1.0',
    description='Tunnel void finder in weak lensing maps & Analysis Toolkit',
    author='Leonardo Maggiore',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-image',
        'astropy',
        'imageio',
        'opencv-python',
        'python-magic',
        'camb',
        'getdist',
        'ipython',
    ],
    include_package_data=True,
)