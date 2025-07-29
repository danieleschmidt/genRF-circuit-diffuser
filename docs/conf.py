# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'GenRF Circuit Diffuser'
copyright = '2025, Daniel Schmidt'
author = 'Daniel Schmidt'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]