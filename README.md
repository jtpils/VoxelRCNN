# carlaok

[TOC]

## Spconv installation tips

* gcc > 5.3
* cmake > 3.13

### GCC Installation

check out [install-gcc-5.4-without-root tutorial](http://www.xieqiang.site/2017/07/31/install-gcc-5.4-without-root/)

### CMake

Download precompiled binaries or build from source (recommend the first option).

If encounter problems when compiling spconv with cmake (e.g. cannot find `CMAKE_ROOT` or cmake was not installed correctly), then change the `setup.py` in the root directory of spconv. `setup.py` use subprocess to call cmake. Change the `cmake` to `/path/to/your/new/gcc/bin/make`