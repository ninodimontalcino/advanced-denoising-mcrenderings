
# ASL Project - Team 053

**Main Paper:** Robust Denoising using Feature and Color Information, Rouselle 
https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12219


 **Team Members**
> - Alexandre Binninger (binninga@student.ethz.ch)
> - Félicité Lordon--de Bonniol du Trémont (tremont@student.ethz.ch)
> - Alexandre Poirrier (apoirrier@student.ethz.ch)
> - Nino Scherrer (ninos@student.ethz.ch)
> 



### Folder Structure

    .
    |── analysis                      # Scripts for automated performance evaluation and plotting
    |── implementations               # Implemented Denoising Algorithms
    │   ├──  basic.cpp                      # Vanilla Implementation of referenced denoising algorithm
    │   ├──  <...>.cpp                      # Optimized Version (mention technique)
    ├── reference_implementation      # Joint Non-Local Means Implementation (Matlab)
    ├── renderings                    # Extracted Buffers of Monte Carlo Renderings
    │   ├──  100spp                         # MC-Rendering using 100 samples per pixel
    |   |    ├── _800x600                            # 800x600 (used for validation)
    │   |    ├── <size>                              # <size> x <size> rendering
    │   ├──  5000spp_GT                      # MC-Rendering using 5000 samples per pixel (Claimed Grountd-Truth)
    ├── src                           # Libraries
    │   ├── ext                             # External Libraries    
    │   |   ├── openexr                             # OpenEXR (used for Data Loading)
    │   |   ├── zlib                                # zlib (data compression used by OpenEXR)
    ├── tests                         # Testing directory
    |   ├──  test_generation              # Modified reference_implementation to create test data
    |   ├──  test_data                    # Generated test data


### Building

```bash
mkdir build
cd build
mkdir temp
git submodule update --init --recursive
cmake ..
make
./main
```

### Tests

To compile tests: 
```bash
mkdir build
cd build
cmake ..
make
cd ..
build/denoise_test
```

In this version, tests are for the FLT function (only the main computation, not the estimators for SURE).
Test data could also be used to compare Matlab OpenEXR and the C++ openEXR implementation.

## Autotuning script

Allows to automatically unroll loops to find best version.

Please place the script and run from build directory. If you want to unroll a file `filename.c`, create another file `filename_unroll.c` next to it, and replace loops to unroll with the unrolling tags (see below).

Usage:

```
python unroll.py ../filename.c n1,n2,n3,...
```

with n1, n2, ... the unrolling factors to test.

Note: best is to keep only one reference function in the `register.hpp` to get faster results.

### Unrolling tags

Use `// $unroll 8` to specify a portion of code to unroll loop (until `// $end_unroll` tag). The unrolling factor is specified in the starting tag.

Then use `$i` for varying indices, and `$n` refers to the unrolling factor.

Example:

```c
// $unroll 2
scalar sum_r_$i;
for(int xp = R; xp < W - R; ++xp) {
    for(int yp = R + F_R; yp < H - R - F_R; yp+=$n) {
        sum_r_$i = 0.f;

        for (int k=-F_R; k<=F_R; k++){
            sum_r_$i += temp[xp * W + yp+k+$i];
        }
        temp2_r[xp * W + yp+$i] = sum_r_$i;
    }
}
// $end_unroll
```

unrolls as

```c
// $unroll 2
scalar sum_r_0;
for(int xp = R; xp < W - R; ++xp) {
    for(int yp = R + F_R; yp < H - R - F_R; yp+=2) {
        sum_r_0 = 0.f;
        sum_r_1 = 0.f;

        for (int k=-F_R; k<=F_R; k++){
            sum_r_0 += temp[xp * W + yp+k+0];
            sum_r_1 += temp[xp * W + yp+k+1];
        }
        temp2_r[xp * W + yp+0] = sum_r_0;
        temp2_r[xp * W + yp+1] = sum_r_1;
    }
}
// $end_unroll
```

Instead of `// $unroll n`, use `// $auto_unroll` for automatically unroll multiple times and compare versions.

