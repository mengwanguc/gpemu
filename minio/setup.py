from distutils.core import setup, Extension

MAJOR = 0
MINOR = 3
MICRO = 3
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name = 'MinIO Cache',
      version = VERSION,
      description = 'MinIO file cache module',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://test.pypi.org/project/MinIO-Cache',
      platforms = "any",
      author = 'Gus Waldspurger',
      author_email = 'gus@waldspurger.com',
      license = 'MIT',
      ext_modules = [
          Extension('minio',
                    sources = [
                        'csrc/miniomodule/miniomodule.c',
                        'csrc/minio/minio.c',
                        'csrc/utils/utils.c'
                    ],
                    extra_link_args = [
                        '-lpthread',
                        '-lrt',
                    ],
                    extra_compile_args = [
                        '-g',
                    ],
                    undef_macros = [
                        "NDEBUG"
                    ])
      ])