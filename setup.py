from setuptools import setup
from setuptools import find_packages

setup(
   name='movie-classifier',
   version='1.0',
   description='CLI movie classification',
   author='Max Xiang',
   author_email='maxx.rift@gmail.com',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   install_requires=['nltk', 'beautifultable', 'inflect'], #external packages
)