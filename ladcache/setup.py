from distutils.core import setup, Extension

MAJOR = 0
MINOR = 0
MICRO = 0
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

with open('README.md', 'r') as f:
    long_description = f.read()

module_ladcache = Extension(
    'ladcache',
    sources = [
        'csrc/ladcachemodule/ladcachemodule.c',
        'csrc/ladcache/cache.c',
        'csrc/utils/alloc.c',
    ],
    extra_link_args = [
        '-lpthread',
        '-luring',
        '-lrt',
    ],
    extra_compile_args = [
        '-g',
        '-Og',
        '-Wno-unused-function',
    ],
    undef_macros = [
        'NDEBUG'
    ],
)

setup(name = 'ladcache',
      version = VERSION,
      description = 'CPython locality-aware distributed cache module',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      platforms = 'Linux',
      author = 'Gus Waldspurger',
      author_email = 'gus@waldspurger.com',
      license = 'MIT',
      ext_modules = [module_ladcache])
