#!/bin/bash

# ----------------------------
# Compile Basic Implementation
# ----------------------------

cd ../build
#cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O0" -DCMAKE_C_FLAGS_RELEASE="-00" 
#cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O3 -fno-vectorize" -DCMAKE_C_FLAGS_RELEASE="-03 -fno-vectorize" 
cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math -march=native" -DCMAKE_C_FLAGS_RELEASE="-03  -ffast-math -march=native" 
make -j4

clear
echo "----------------------------------------------"
echo " PERFORMANCE EVALUATION - Basic Implementation"
echo "----------------------------------------------"

# -------------------------------------------------------
# Process different image resolutions in increasing order
# -------------------------------------------------------
output="["

for img_size in {64,80,96,112,128,256,512,1024}                     # ENABLE LINE FOR SPECIFIC EVALUATION
#for img_size in `ls ../renderings/100spp/ | sort -V`;              # ENABLE LINE FOR EVALUATION OF ALL RESOLUTIONS
do
    printf $img_size
    printf "x"
    printf $img_size;
    printf ": "

    # Process for fixed R=10
    CYCLES=`./main $img_size 10 |grep cycles |rev | cut -c8- |rev`
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




