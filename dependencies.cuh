#ifndef DEPENDENCIES_CUH_
#define DEPENDENCIES_CUH_
#include <iostream>
#include <algorithm>
#include <exception>
#include <random>
#include <iostream>
#include <ctime>
#include "math.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <thread>
#include <future>
#include <map>
#ifdef WINDOWS
#include <windows.h>
#include <winbase.h>
#endif
const double one = 1, zero = 0;

#include "mkl.h"
#include "mkl_lapack.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif