# Combinatorial-GMDH
It's a bit messy, because I've first tried to make it pretty, flexible and conveinient to use, but time it took to run on CUDA has cooled me off. Now it runs considerably faster, but still slower than on CPU (at least with the data I've tried).
The memory ops are as optimal as I could make them in a week, and GPU utilization is near 70% (GeForce 940MX).
Should use multiple GPUs if present, but I haven't tested that.

It finds model of optimal complexity via combinatorial algorithm of group methods of data handling class.
If built, runs with CUDA or fully on CPU if `cpu` argument is specified.

Takes csv files on input, specify paths to them as follows `x <path to x.csv> y <path to y.csv>`.
CPU mode also supports `verbose` setting, which, if enabled, will produce detailed output about each candidate-model.
Number of threads is specified by `nt`. 

Tested on Windows, built with Visual Studio, CMakeLists serves as decoration, mainly, but it's not completely impossible that it works.
