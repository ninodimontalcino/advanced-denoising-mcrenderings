
# ASL Project - Team 053

**Main Paper:** Robust Denoising using Feature and Color Information, Rouselle 
https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12219


 **Team Members**
> - Alexandre Binning (binninga@student.ethz.ch)
> - Félicité Lordon--de Bonniol du Trémont (ftremont@student.ethz.ch)
> - Alexandre Poirrier (apoirrier@student.ethz.ch)
> - Nino Scherrer (ninos@student.ethz.ch)
> 



### Folder Structure

    .
    |── implementations               # Implemented Denoising Algorithms
    │   ├──  basic.cpp                      # Vanilla Implementation of referenced denoising algorithm
    │   ├──  <...>.cpp                      # Optimized Version (mention technique)
    ├── reference_implementation      # Joint Non-Local Means Implementation (Matlab)
    ├── renderings                    # Extracted Buffers of Monte Carlo Renderings
    │   ├──  100spp                         # MC-Rendering using 100 samples per pixel
    |   |    ├── _800x600                            # 800x600 (used for validation)
    │   |    ├── 256                                 # 256x256
    │   |    ├── 512                                 # 512x512
    │   |    ├── 1024                                # 1024x1024
    │   |    ├── 2048                                # 2048x2048
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
