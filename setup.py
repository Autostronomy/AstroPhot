from setuptools import setup
import autoprof as ap
setup(
    name="autoprof",
    version=ap.__version__,    
    description="Fast and flexible astronomical image modelling tool",
    url="https://github.com/ConnorStoneAstro/AutoProf-2",
    author=ap.__author__,
    author_email=ap.__email__,
    license="GPL-3.0 license",
    packages=["autoprof"],
    install_requires=["scipy",
                      "numpy",
                      "astropy",
                      "matplotlib",
                      ],
    entry_points = {
        'console_scripts': [
            'autoprof = autoprof:run_from_commandline',
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  
        "Programming Language :: Python :: 3",
    ],
)
