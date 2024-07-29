from setuptools import setup, find_packages

setup(
    name='gpemu',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    test_suite='tests',
    author='Meng Wang',
    author_email='mengwanguc@gmail.com',
    description='GPEmu: a GPU emulator for faster and cheaper prototyping and evaluation' +
                ' of deep learning system research',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mengwanguc/gpemu',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
