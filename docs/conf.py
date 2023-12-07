#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import astrophot as ap

# Add the package root to the system path to enable autodoc to find modules.
sys.path.insert(0, os.path.abspath("../"))

# -- General configuration ------------------------------------------------

# Extensions to use
extensions = [
    'nbsphinx',
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    'sphinx.ext.viewcode',
]

# Paths to templates
templates_path = ["_templates"]

# Suffixes of source filenames
source_suffix = ".rst"

# Master document
master_doc = "index"

# Project information
project = "AstroPhot"
copyright = "2023, Connor Stone"
author = "Connor Stone"

# Version information
version = ap.__version__
release = ap.__version__

# Patterns of files and directories to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Pygments style to use
pygments_style = "sphinx"

# Whether to include TODOs
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# Theme to use
html_theme = "sphinx_rtd_theme"

# Sidebar templates
html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
    ]
}

html_favicon = "media/AP_logo_favicon.ico"

# Output file base name for HTML help builder
htmlhelp_basename = "AstroPhotdocs"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# LaTeX document settings
latex_documents = [
    (master_doc, "AstroPhot.tex", "AstroPhot Documentation", "Connor Stone", "manual"),
]

# -- Options for manual page output ---------------------------------------

man_pages = [
    (master_doc, "astrophot", "AstroPhot Documentation", [author], 1),
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "AstroPhot",
        "AstroPhot Documentation",
        author,
        "AstroPhot",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for nbsphinx --------------------------------------------------
nbsphinx_execute = 'never'
