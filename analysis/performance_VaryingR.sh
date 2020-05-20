#!/bin/bash

# ----------------------------------------------------
# Compile Basic Implementation with -O3 -fno-vectorize
# ----------------------------------------------------

cd ../build
#cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O0" -DCMAKE_C_FLAGS_RELEASE="-00" 
#cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O3 -fno-vectorize" -DCMAKE_C_FLAGS_RELEASE="-03 -fno-vectorize" 
#cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math -march=native -fno-tree-vectorize" -DCMAKE_C_FLAGS_RELEASE="-03 -ffast-math -march=native -fno-tree-vectorize" 
cmake .. 
make -j4

clear
echo "-------------------------------------------------"
echo " PERFORMANCE EVALUATION - for varying window size"
echo "--------------------------------------------------"

# -------------------------------------------------------
# Process different image resolutions in increasing order
# -------------------------------------------------------

for img_size in {128,256,512,1024}  
#for img_size in `ls ../renderings/100spp/ | sort -V`; 
do
    echo "=========================================="
    echo "Processing:" $img_size"x"$img_size;
    echo "------------------------------------------"

    output="["

    # Process for varying values of R
    for value in {3,5,7,9,11,13,15};
    do
        printf "r="$value": "
        CYCLES=`./main $img_size $value |grep cycles |rev | cut -c8- |rev`
        echo $CYCLES
        output+=$CYCLES
        output+=", "
    done

    # Output array for plotting
    output="${output:0:${#output}-2}"
    output+="]"
    printf "\n"
    echo "=> Array for Plotting:"
    echo $output    

done