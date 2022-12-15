from setuputils import setup

setup(
    name="autoprof",
    version="0.1.0",    
    description="Fast and flexible astronomical image modelling tool",
    url="https://github.com/ConnorStoneAstro/AutoProf-2",
    author="Connor Stone",
    author_email="connorstone628@gmail.com",
    license="GPL-3.0 license",
    packages=["autoprof"],
    install_requires=["scipy",
                      "numpy",
                      "torch",
                      "astropy",
                      ],
    entry_points = [
        'console_scripts': [
            'autoprof = autoprof:run_from_commandline',
        ],
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL-3.0 license",  
        "Operating System :: POSIX :: Linux",        
        "Programming Language :: Python :: 3",
    ],
)
