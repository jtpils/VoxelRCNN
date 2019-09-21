# CARLA-OK!

[TOC]

## Installations

Prerequisites: spconv, etc.

Test under: Ubuntu 18, torch=1.2.0, python=3.7

### spconv

* gcc >= 5.3, < 8 if nvcc <= 10.1
* cmake > 3.13

#### GCC Installation

check out [install-gcc-5.4-without-root tutorial](http://www.xieqiang.site/2017/07/31/install-gcc-5.4-without-root/)

Download from [gcc releases mirrors](https://bigsearcher.com/mirrors/gcc/releases/)

#### CMake

Download precompiled binaries or build from source (recommend the first option).

If encounter problems when compiling spconv with cmake (e.g. cannot find `CMAKE_ROOT` or cmake was not installed correctly), then change the `setup.py` in the root directory of spconv. `setup.py` use subprocess to call cmake. Change the `cmake` to `/path/to/your/new/gcc/bin/make`

