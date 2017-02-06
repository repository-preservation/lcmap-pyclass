# Pyclass - Python implementation of the classification methodology used for land cover on the LCMAP project.

## Using pyclass
```python
>>> import pyclass
>>> model, random_seed = pyclass.train(trends, coefs, rmse, dem, aspect, slope, posidex, mpw, qa, random_seed)
>>>
>>> type(model)
<class 'sklearn.ensemble.forest.RandomForestClassifier'>
>>>
>>> classes, probabilities = pyclass.classify(model, coefs[0], rmse[0], dem[0], aspect[0], slope[0], posidex[0], mpw[0], qa[0])
>>>
>>> classes
[  0.   1.   2.   5.   6.   7.   8.   9.  10.]
>>>
>>> probabilities
[[ 0.1   0.    0.1   0.01  0.09  0.36  0.16  0.03  0.15]]
>>>
>>> classes.take(np.argmax(probabilities))
6.0
```

## Installing
System requirements (Ubuntu)
* python3-dev
* gfortran
* libopenblas-dev
* liblapack-dev
* graphviz
* python-virtualenv

System requirements (Centos)
* python3-devel
* gfortran
* blas-dev
* lapack-dev
* graphviz
* python-virtualenv

It's highly recommended to do all your development & testing in a virtual environment.
```bash
user@dev:/home/user/$ mkdir pyclass
user@dev:/home/user/$ cd pyclass
user@dev:/home/user/pyccd$ virtualenv -p python3 .venv
user@dev:/home/user/pyccd$ . .venv/bin/activate
(.venv) user@dev:/home/user/pyclass$
```

The rest of the command prompts are truncated to ```$``` for readability, but assume an activated virtual environment and pwd as above, or that you know what you are doing.

##### Clone the repo
```bash
$ git clone https://github.com/usgs-eros/lcmap-pyclass.git
```
or if you have ssh keys set up in github:
```bash
$ git clone git@github.com:usgs-eros/lcmap-pyclass.git
```

##### Install dev dependencies
Install jupyter notebook.
```bash
$ pip install -e .[dev]
```

##### Install test dependencies
```bash
$ pip install -e .[test]
```

## Testing & Running
```bash
$ pytest
$ pytest --profile
$ pytest --profile-svg

# pytest-watch
$ ptw
```

## Contributing
Contributions to pyccd are most welcome, just be sure to thoroughly review the guidelines first.

[Contributing](docs/CONTRIBUTING.md)

[Developers Guide](docs/DEVELOPING.md)

## Versions
Pyclass versions comply with [PEP440](https://www.python.org/dev/peps/pep-0440/)
and [Semantic Versioning](http://semver.org/), thus MAJOR.MINOR.PATCH.LABEL as
defined by:

> Given a version number MAJOR.MINOR.PATCH, increment the:

> 1. MAJOR version when you make incompatible API changes

> 2. MINOR version when you add functionality in a backwards-compatible manner, and

> 3. PATCH version when you make backwards-compatible bug fixes.

> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Alpha releases (x.x.x.ax) indicate that the code functions but the result may
or may not be correct.

Beta releases (x.x.x.bx) indicate that the code functions and the results
are believed to be correct by the developers but have not yet been verified.

Release candidates (x.x.x.rcx) indicate that the code functions and the results
are correct according to the developers and verifiers and is ready for final
performance and acceptance testing.

Full version releases (x.x.x) indicate that the code functions, the results
are verified to be correct and it has passed all testing and quality checks.

Pyclass's version is defined by the ```pyclass/version.py/__version__``` attribute
ONLY.
