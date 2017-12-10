# Combinatorial-GMDH
It's a bit messy, as I've first intendet to make it flexible and conveinient to use, but time it took to run on CUDA cooled me off. Now it runs considerably faster, but still slower than on CPU (on my laptop's GeForce 940MX, though).
The memory ops are as optimal as I could make them in a week, and GPU utilization is near 70%.

~~Should use multiple GPUs if present, but I haven't tested that.~~ Actually, it doesn't wrok that well. The reason for this is pretty obvious, when you have to deal with it: matrices of input data get coppied only into device 0 memory space. I've made a hotfix wich allows to find the best mask, but breaks the computation of unbiasedness criterion. Since I don't currently have time to fix it fully, I encourage you to use single-GPU mode for now (you are always wellcome do a pull request with a fix).

It finds model of optimal complexity via combinatorial algorithm of group methods of data handling class.
If built, runs with CUDA or fully on CPU if `cpu` argument is specified.

Takes csv files on input, specify paths to them as follows `x <path to x.csv> y <path to y.csv>`.
CPU mode also supports `verbose` setting, which, if enabled, will produce detailed output about each candidate-model.
Number of threads is specified by `nt`. 

Tested on Windows, built with Visual Studio, CMakeLists serves as decoration, mainly, but it's not completely impossible that it works.
