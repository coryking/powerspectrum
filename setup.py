import setuptools

setuptools.setup(
    name = "freqency",
    version = "0.0.4",
    author = "Cory",
    descripton='Frequency tools',
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['freq'],
    install_requires=['numpy', 'matplotlib', 'scikits.audiolab'],
    entry_points={
            'console_scripts': [
                'plot-spectrum=freq.freq:cli',
            ]
    },

)