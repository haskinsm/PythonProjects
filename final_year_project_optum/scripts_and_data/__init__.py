# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:30:53 2021

@author: micha


The __init__.py files are required to make Python treat directories containing the file as packages.
This prevents directories with a common name, such as string, unintentionally hiding valid modules that 
occur later on the module search path. In the simplest case, __init__.py can just be an empty file, but
it can also execute initialization code for the package or set the __all__ variable, described later.

src: https://realpython.com/python-import/
"""

from . import scripts
from . import data