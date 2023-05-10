#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Add the package root to the system path to enable autodoc to find modules.
sys.path.insert(0, os.path.abspath("../"))

# -- General configuration ------------------------------------------------

# Extensions to use
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Paths to templates
templates_path = ["_templates"]

# Suffixes of source filenames
source_suffix = ".rst"

# Master document
master_doc = "index"

# Project information
project = "AutoProf"
copyright = "2023, Connor Stone"
author = "Connor Stone"

# Version information
version = "0.7"
release = "0.7.2"

# Patterns of files and directories to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
htmlhelp_basename = "AutoProfdocs"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# LaTeX document settings
latex_documents = [
    (master_doc, "AutoProf.tex", "AutoProf Documentation", "Connor Stone", "manual"),
]

# -- Options for manual page output ---------------------------------------

man_pages = [
    (master_doc, "autoprof", "AutoProf Documentation", [author], 1),
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "AutoProf",
        "AutoProf Documentation",
        author,
        "AutoProf",
        "One line description of project.",
        "Miscellaneous",
    ),
]
