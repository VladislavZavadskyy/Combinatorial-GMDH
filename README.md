# Combinatorial-GMDH
Finds model of optimal complexity via combinatorial algorithm of group methods of data handling class.
If built, runs with CUDA or fully on CPU if `cpu` argument is specified.

Takes csv files on input, specify paths to them as follows `x <path to x.csv> y <path to y.csv>`.
CPU mode also supports `verbose` setting, which, if enabled, will produce detailed output about each candidate-model.
Number of threads is specified by `nt`. 

Tested on Windows, built with Visual Studio, CMakeLists serves as decoration, mainly, but it's not completely impossible that it works.
