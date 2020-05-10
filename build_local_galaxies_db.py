#!/usr/bin/env python

""" Builds the local_galaxies database.
    17.11.2017, tgreve@ucl.ac.uk
"""

from importlib import reload
import logal
reload(logal)
from logal import galaxies_db
reload(logal.galaxies_db)

galaxies_db.build()
