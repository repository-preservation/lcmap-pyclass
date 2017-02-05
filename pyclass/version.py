""" Module specifically to hold algorithm version information.  The reason this
exists is the version information could be needed in both setup.py for install and
also in subcomponents.

Do not import anything into this module."""
__version__ = '1.0.0.b1'
__name = 'lcmap-pyclass'
__algorithm__ = ':'.join([__name, __version__])
