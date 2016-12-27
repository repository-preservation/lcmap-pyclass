""" Main bootstrap and configuration module for pyccd.  Any module that
requires configuration or services should import app and obtain the
configuration or service from here.
app.py enables a very basic but sufficient form of loose coupling
by setting names of services & configuration once and allowing other modules
that require these services/information to obtain them by name rather than
directly importing or instantiating.
Module level constructs are only evaluated once in a Python application's
lifecycle, usually at the time of first import. This pattern is borrowed
from Flask.
"""
import logging
import sys


############################
# Logging system
############################
# to use the logging from any module:
# import app
# logger = app.logging.getLogger(__name__)
#
# To alter where log messages go or how they are represented,
# configure the
# logging system below.
# iso8601 date format
#__format = '%(asctime)s %(module)s::%(funcName)-20s - %(message)s'
__format = '%(asctime)s.%(msecs)03d %(module)s::%(funcName)-20s - %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format=__format,
                    datefmt='%Y-%m-%d %H:%M:%S')
