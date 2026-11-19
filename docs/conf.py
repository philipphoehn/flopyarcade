"""Sphinx configuration for FloPyArcade documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'FloPyArcade'
author = 'Philipp Hoen'
version = '0.1'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
html_static_path = []

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
