# dea-netcdf-benchmark

Prototype implementation of the parallel netcdf reader (Python based) and benchmarking for it.

> As of right now it's not a usable library, but moving in that direction rapidly. 
> Preliminary benchmarks show significant improvement in throughput compared to single threaded reads.

## Problem Definition

DEA is storing large amounts of Earth observation data and derived products in netcdf files on the NCI managed infrastructure. Underlying storage system, Lustre, is quite capable of scaling with more reading threads, also significant part of the load cost is in decompression which also scales well as you throw more processing threads at it. As such we expect significant throughput gains by performing data load in parallel. Unfortunately underlying IO library `HDF5` is not capable of multi-threaded operation in non-MPI application. While MPI is great for large scale processing it is too much of a burden to setup to use in a simple data analysis notebook.

## Solution

Use multiple processes, shared memory and standard python IPC libraries to read netcdf files in parallel while hiding the complexity of multi-process management from the end user as much as practical.
