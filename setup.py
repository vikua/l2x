import os 
import sys 
from setuptools import setup, find_packages


PY_VER = sys.version_info 


if not PY_VER >= (3, 5): 
    raise RuntimeError('mlserve does not support Python earlier than 3.6')


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


setup(
    name='l2x', 
    description='Model interpretation toolkit based on Learning to Explain', 
    long_description=read('README.md'), 

    author='Viktor Kovryzhkin',
    author_email='vik.kovrizhkin@gmail.com',

    platforms=['POSIX'],

    url='https://github.com/vikua/l2x.git',
    license='MIT', 
    keywords='l2x neural prediction explanation',

    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=[
        'tensorflow',
        'pandas',
    ],

    extras_require={
        'test': ['pytest'],
    }
)   