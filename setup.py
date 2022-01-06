from setuptools import setup, find_packages

RAY_VERSION = "1.8.0"
DASK_VERSION = "2021.9.1"

setup(
    name="qpl",
    version="0.1",
    description="",
    author="Joe Hurley",
    author_email="jhurley@figure.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
       # "cac_calibrate",
        #"PyYAML",
        #f"dask[complete]=={DASK_VERSION}",
        #"findspark>=1.4.2",
        "gcsfs>=2021.4.0",
        "google-cloud-bigquery[bqstorage,pandas]>=2.15.0",
        "google-cloud-storage",
        #"hyperopt>=0.2.5",
        #"lightgbm>=3.1.1",
        #"lmdb",
        "numpy",
        #"nvidia-ml-py3",
        "openpyxl",
        "pandas>=1.0.1",
        #"pqbqjoin",
        #"prefetch_generator",
        #"pyspark>=3.1.1",
        "pytest",
        "python-snappy",
        #f"ray=={RAY_VERSION}",
        #f"ray[default]=={RAY_VERSION}",
        #f"ray[tune]=={RAY_VERSION}",
        "scikit_learn>=0.24.1",
        "tables",
        #"torch==1.9.0",
        #"torchmetrics",
        "vm-utils",
        "xgboost_ray",
        "xlrd"
    ],
    zip_safe=False
)
