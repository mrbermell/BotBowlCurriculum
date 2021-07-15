from setuptools import setup, find_packages

setup(name='botbowlcurriculum',
      version="0.1",
      include_package_data=True,
      install_requires=[
          'numpy',
          'statsmodels'
      ],
      packages=find_packages()
)
