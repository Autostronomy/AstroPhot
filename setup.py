from setuptools import setup, find_packages
import autoprof.__init__ as ap
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

print(read("README.md"))
setup(
    name="autoprof",
    version=ap.__version__,    
    description="A fast, flexible, and automated astronomical image modelling tool for precise parallel multi-wavelength photometry",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/ConnorStoneAstro/AutoProf",
    author=ap.__author__,
    author_email=ap.__email__,
    license="GPL-3.0 license",
    packages=find_packages(),
    install_requires=["scipy",
                      "numpy",
                      "astropy",
                      "matplotlib",
                      "torch",
                      ],
    entry_points = {
        'console_scripts': [
            'autoprof = autoprof:run_from_terminal',
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  
        "Programming Language :: Python :: 3",
    ],
)
