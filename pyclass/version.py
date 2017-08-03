""" Module specifically to hold algorithm version information.  The reason this
exists is the version information could be needed in both setup.py for install and
also in subcomponents.

Do not import anything into this module."""
__version__ = '2017.08.03'
__name = 'lcmap-pyclass'
__algorithm__ = ':'.join([__name, __version__])
