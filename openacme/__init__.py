__all__ = ['OPENACME_BASE']
__version__ = '0.1.0'

import logging

import pystow

OPENACME_BASE = pystow.module('openacme')

logging.basicConfig(format=('%(levelname)s: [%(asctime)s] %(name)s'
                            ' - %(message)s'),
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
