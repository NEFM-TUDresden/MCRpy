"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import pickle
import shutil
import time
import subprocess
from typing import List, Dict

import numpy as np


def setup_logging(target_folder: str, args: argparse.Namespace):
    """Set up logging."""
    logging_format = '%(asctime)s on %(levelname)s: %(message)s'
    if target_folder is None:
        logging.basicConfig(format=logging_format, level=args.logging_level)
    else:
        logfile_additives = '-' + time.asctime().replace(
                ' ', '-').replace(':', '-') if args.logfile_date else ''
        logging_filename = '{}{}.log'.format(args.logfile_name, logfile_additives)
        logging_filepath = os.path.join(target_folder, logging_filename)
        logging.basicConfig(filename=logging_filepath,
                            format=logging_format, level=args.logging_level)

def log_this(function):
    """Decorator for logging a function without much boilerplate.
    Logs time when function is entered an exited to logging.info.
    Logs all exceptionsto logging.exception.
    Logs args on entering to logging.debug.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            infostring = '>> {} entered'.format(function.__name__)
            logging.info(infostring)
            if logging.root.level <= logging.DEBUG:
                debugstring = \
                    '        with args = {}'.format(args)
                debugstring = debugstring + os.linesep + \
                    '        and kwargs = {}'.format(kwargs)
                logging.debug(debugstring)
            result = function(*args, **kwargs)
            logging.info('<< {} exited'.format(function.__name__))
            return result
        except:
            exceptionstring = 'Exception in function {}'.format(
                function.__name__)
            exceptionstring = exceptionstring + os.linesep + \
                '        with *args = {}'.format(args)
            exceptionstring = exceptionstring + os.linesep + \
                '        and *kwargs = {}'.format(kwargs)
            logging.exception(exceptionstring)
            raise
    return wrapper


