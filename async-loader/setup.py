from distutils.core import setup, Extension

MAJOR = 0
MINOR = 2
MICRO = 0
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

with open('README.md', 'r') as f:
    long_description = f.read()

module_asyncloader = Extension(
    'AsyncLoader',
    sources = [
        'csrc/asyncmodule/asyncmodule.c',
        'csrc/async/async.c',
        'csrc/utils/alloc.c',
        'csrc/utils/sort.c',
    ],
    extra_link_args = [
        '-lpthread',
        '-luring',
        '-lrt',
    ],
    extra_compile_args = [
        '-g',
    ],
    undef_macros = [
        "NDEBUG"
    ],
)

setup(name = 'Async File Loader',
      version = VERSION,
      description = 'Python asynchronous file loader module',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      platforms = "any",
      author = 'Gus Waldspurger',
      author_email = 'gus@waldspurger.com',
      license = 'MIT',
      ext_modules = [module_asyncloader])