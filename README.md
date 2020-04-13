
# ASL Project - Team 053

**Main Paper:** Robust Denoising using Feature and Color Information, Rouselle 
https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12219


 **Team Members**
> - Alexandre Binning (binninga@student.ethz.ch)
> - Félicité Lordon--de Bonniol du Trémont (tremont@student.ethz.ch)
> - Alexandre Poirrier (apoirrier@student.ethz.ch)
> - Nino Scherrer (ninos@student.ethz.ch)
> 



### Folder Structure

    .
    ├── reference_implementation      # Joint Non-Local Means Implementation (Matlab)
    ├── renderings                    # Extracted Buffers of Monte Carlo Renderings
    │   ├──  100spp                       # MC-Rendering using 100 samples per pixel
    │   ├──  500spp_GT                    # MC-Rendering using 500 samples per pixel (Claimed Grountd-Truth)
    ├── src/ext                       # Openexr and zlib repositories
    ├── tests                         # Testing directory
    |   ├──  test_generation              # Modified reference_implementation to create test data
    |   ├──  test_data                    # Generated test data


### EXR Loading

mkdir build
cd build
git submodule update --init --recursive
cmake ..
make
./exe

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
