# Matrix Multiply

## Overview
倍精度浮動小数点数における1024x1024型の行列積の計算のベンチマークを計ってみた.

### Description

とりあえず、以下の最適化を試してみました.

* Normal
* Cache
* OpenMP
* intel AVX
* AVX + LoopUnroll
* AVX + LoopUnroll + Cache
* AVX + LoopUnroll + Cache + OpenMP
* nVidia Cuda


And open source with a [public repository][mnrn] on GitHub.

### Demo

それぞれ5回ほど試して得られた平均ベンチマークは以下のようになってます。

* Normal      315.40220(ms)
* Cache      112.94790(ms)
* OpenMP[parallelism]  87.04746(ms)
* avx[SIMD]   65.76156(ms)
* avx+LoopUnroll 38.81741(ms)
* avx+LoopUnroll+Cache 18.001544(ms)
* avx+LoopUnroll+Cache+OpenMP 6.146638(ms)
* Cuda[GPGPU]     1.16271(ms)

![benchmark](https://github.com/mnrn/dgemm/blob/master/data/benchmark2.png"benchmark")

### Requirement

requires [gcc](https://gcc.gnu.org/) v5+ ,  [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) v8+ and [GNUmake](https://www.gnu.org/software/make/) to run.


### Usage

```sh
$ cd avx
$ make
$ ./bin/matrix
```

License
----

Public Domain


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [mnrn]: <https://github.com/mnrn/dgemm>
   
   [gcc]: <https://gcc.gnu.org/>
   [cuda-toolkit]: <https://developer.nvidia.com/cuda-toolkit>
   [GNUmake]: <https://www.gnu.org/software/make/>

