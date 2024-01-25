# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ICE'
copyright = '2023, ICE Team'
author = 'ICE Team'
#release = '0'

#--Path setup-----------------------------------------------------------------


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
   "logo": {
      "image_light": "_static/logo-light.png",
      "image_dark": "_static/logo-dark.png",
   }
}


# -- Myst-nb configuration ---------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_mode = 'off'