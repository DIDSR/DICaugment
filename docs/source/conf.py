# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# from sphinx.ext.autosummary.generate import AutosummaryRenderer

sys.path.insert(0, os.path.abspath("../.."))




project = 'Albumentations3D'
copyright = '2023, Jacob McIntosh'
author = 'Jacob McIntosh'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc",]

templates_path = ['_templates']
exclude_patterns = []

# autosummary_generate = True
add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



# def smart_fullname(fullname):
#     parts = fullname.split(".")
#     return ".".join(parts[1:])


# def fixed_init(self, app):
#     AutosummaryRenderer.__old_init__(self, app)
#     self.env.filters["smart_fullname"] = smart_fullname


# AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
# AutosummaryRenderer.__init__ = fixed_init