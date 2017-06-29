"""pyclass is an implementation of the Classification Methodology
outlined for inclusion into the LCMAP project.

Principal Algorithm investigator:

Zhe Zhu
Assistant Professor,
Department of Geosciences,
Texas Tech University, TX, USA
"""

from setuptools import setup
from os import path
import io

here = path.abspath(path.dirname(__file__))


# bring in __version__ and __name from version.py for install.
with open(path.join(here, 'pyclass', 'version.py')) as h:
    exec(h.read())

setup(

    # __name is defined in version.py
    name=__name,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    # __version__ is defined in version.py
    version=__version__,
    
    description='Python implementation of LCMAP Classification',
    long_description=__doc__,
    url='https://github.com/usgs-eros/lcmap-pyclass',
    maintainer='klsmith-usgs',
    maintainer_email='kelcy.smith.ctr@usgs.gov',
    license='Public Domain',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: Public Domain',

        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='python change detection',

    packages=['pyclass'],

    install_requires=['numpy>=1.13.0',
                      'scipy>=0.18.1',
                      'scikit-learn>=0.18',
                      'PyYAML>=3.12'],

    extras_require={
        'test': ['flake8>=3.0.4',
                 'coverage>=4.2',
                 'pytest>=3.0.2',
                 'pytest-profiling>=1.1.1',
                 'gprof2dot>=2015.12.1',
                 'pytest-watch>=4.1.0'],
        'dev': ['jupyter',],
    },

    setup_requires=['pytest-runner', 'pip'],
    tests_require=['pytest>=3.0.2'],

    package_data={
        'pyclass': ['parameters.yaml'],
    }
)
