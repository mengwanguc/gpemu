from distutils.core import setup, Extension

MAJOR = 0
MINOR = 0
MICRO = 0
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name = 'mlock',
      version = VERSION,
      description = 'page-locked memory interface for Python',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      platforms = "any",
      author = 'Gus Waldspurger',
      author_email = 'gus@waldspurger.com',
      license = 'MIT',
      ext_modules = [
            Extension('mlock', sources = ['csrc/mlockmodule/mlockmodule.c'])
      ])