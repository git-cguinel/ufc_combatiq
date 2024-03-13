from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='combat_iq',
      description="Demo Day Project : UFC Combatiq",
      install_requires=requirements,
      packages=find_packages())
