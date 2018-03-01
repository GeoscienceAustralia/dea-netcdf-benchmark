from setuptools import setup, find_packages


setup(
    name='mpnetcdf4',
    version='0.0.1',
    python_requires='>=3.5.2',
    url='https://github.com/GeoscienceAustralia/dea-netcdf-benchmark',
    description='Exploration into improving NetCDF IO at scale using DEA/ODC',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],

    packages=find_packages(),
    install_requires=[
        'numpy',
        'h5py',
        'rasterio',
    ],
    tests_require=[
        'pyyaml',
    ],
)
