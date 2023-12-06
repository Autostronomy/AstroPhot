from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="astrophot",
    version=read("astrophot/VERSION"),    
    description="A fast, flexible, differentiable, and automated astronomical image modelling tool for precise parallel multi-wavelength photometry",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/Autostronomy/AstroPhot",
    author="Connor Stone",
    author_email="connorstone628@gmail.com",
    license="GPL-3.0 license",
    packages=find_packages(),
    package_data={'astrophot':['VERSION']},
    install_requires=["scipy",
                      "numpy",
                      "astropy",
                      "matplotlib",
                      "torch",
                      "tqdm",
                      "requests",
                      "h5py",
                      "pyyaml",
                      "pyro-ppl",
                      ],
    entry_points = {
        'console_scripts': [
            'astrophot = astrophot:run_from_terminal',
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  
        "Programming Language :: Python :: 3",
    ],
)
