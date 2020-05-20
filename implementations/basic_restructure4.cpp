#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
#include "../flt_restructure.hpp"
#include "../memory_mgmt.hpp"

#include "../denoise.h"
#include <immintrin.h>
// #include "../avx_mathfun.h"


using namespace std;

/*! -------------------------------------------------------
	Vanilla Denoising Algorithm (without any optimization)
    
    Parameters:
        - out_img (buffer)      Destination buffer for final denoised image
        - c (buffer)            Color Buffer
	    - c_var (buffer) 	    Variance Buffer of Color
	    - f	(buffer)            Feature Buffer  
                                    f[0] := albedo
                                    f[1] := depth
                                    f[2] := normal
        - f_var (buffer)	    Variance Buffer of Features
        - R (int)               Window radius
        - img_width (int)       Image Width
        - img_height (int)      Image Height
å
    \return void --> denoised image in buffer out_img
 */
 void basic_restructure4(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height){

    if(DEBUG) {
        cout << "--------------------------------------------------" << endl;
        cout << " Starting Algorithm " << endl;
    }

    // ----------------------------------------------
    // (1) Precomputation A
    // ----------------------------------------------
    

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    Flt_parameters p_pre = { .kc = 1., .kf = INFINITY, .tau = 0., .f = 3, .r = 5};
    buffer f_filtered;
    buffer f_var_filtered;
    allocate_buffer_zero(&f_filtered, img_width, img_height);
    allocate_buffer_zero(&f_var_filtered, img_width, img_height);
    feature_prefiltering_VEC(f_filtered, f_var_filtered, f, f_var, p_pre, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/albedo_filtered.exr", &f_filtered[0], img_width, img_height);
        write_channel_exr("temp/depth_filtered.exr",  &f_filtered[1], img_width, img_height);
        write_channel_exr("temp/normal_filtered.exr", &f_filtered[2], img_width, img_height);
        cout << "\t - Feature Prefiltering done" << endl;
    }

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------

    // (a) Candidate Filter: FIRST
    Flt_parameters p_all[3];
    p_all[0] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    p_all[1] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    p_all[2] = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};   // Fixed Variable: kc=INF => is exploited in filtering
    buffer r, g, b;
    allocate_buffer_zero(&r, img_width, img_height);
    allocate_buffer_zero(&g, img_width, img_height);
    allocate_buffer_zero(&b, img_width, img_height);
    candidate_filtering_all_VEC(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, img_width, img_height);

    // candidate_filtering_FIRST_VEC(r, c, c_var, f_filtered, f_var_filtered, p_all[0], img_width, img_height);
    // candidate_filtering_SECOND_VEC(g, c, c_var, f_filtered, f_var_filtered, p_all[1], img_width, img_height);
    // candidate_filtering_THIRD_VEC(b, c, c_var, f_filtered, f_var_filtered, p_all[2], img_width, img_height);
    

    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST.exr", &r, img_width, img_height);
        write_buffer_exr("temp/candidate_SECOND.exr", &g, img_width, img_height);
        write_buffer_exr("temp/candidate_THIRD.exr", &b, img_width, img_height);
        cout << "\t - Candidates done" << endl;
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    buffer sure;
    allocate_buffer(&sure, img_width, img_height); // Zero allocation done in Sure
    sure_all_VEC(sure, c, c_var, r, g, b, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/sure_r.exr", &sure[0], img_width, img_height);
        write_channel_exr("temp/sure_g.exr", &sure[1], img_width, img_height);
        write_channel_exr("temp/sure_b.exr", &sure[2], img_width, img_height);   
        cout << "\t - Sure Error Estimates done" << endl;
    }

    // (b) Filter error estimates
    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    buffer e;
    allocate_buffer_zero(&e, img_width, img_height);
    filtering_basic_f1_VEC(e, sure, c, c_var, p_sure, img_width, img_height);
    
    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/e_r.exr", &e[0], img_width, img_height);
        write_channel_exr("temp/e_g.exr", &e[1], img_width, img_height);
        write_channel_exr("temp/e_b.exr", &e[2], img_width, img_height);
        cout << "\t - Filtered Sure Error Estimates done" << endl;
    }

     // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    buffer sel;
    allocate_buffer(&sel, img_width, img_height);
    
    __m256 e0, e1, e2;
    __m256 mask0_0, mask1_0, mask_0;
    __m256 mask0_1, mask1_1, mask_1;
    __m256 mask0_2, mask1_2, mask_2;
    const __m256 ones = _mm256_set1_ps(1.);
    // Compute selection maps
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y+=8){

                e0 = _mm256_loadu_ps(e[0][x]+y);
                e1 = _mm256_loadu_ps(e[1][x]+y);
                e2 = _mm256_loadu_ps(e[2][x]+y);
                
                mask0_0 = _mm256_cmp_ps(e0, e1, _CMP_LT_OQ);
                mask1_0 = _mm256_cmp_ps(e0, e2, _CMP_LT_OQ);
                mask0_1 = _mm256_cmp_ps(e1, e0, _CMP_LT_OQ);
                mask1_1 = _mm256_cmp_ps(e1, e2, _CMP_LT_OQ);
                mask0_2 = _mm256_cmp_ps(e2, e0, _CMP_LT_OQ);
                mask1_2 = _mm256_cmp_ps(e1, e2, _CMP_LT_OQ);

                mask_0 = _mm256_and_ps(mask0_0, mask1_0);
                mask_1 = _mm256_and_ps(mask0_1, mask1_1);
                mask_2 = _mm256_and_ps(mask0_2, mask1_2);
                mask_0 = _mm256_and_ps(ones, mask_0);
                mask_1 = _mm256_and_ps(ones, mask_1);
                mask_2 = _mm256_and_ps(ones, mask_2);
                _mm256_storeu_ps(sel[0][x]+y, mask_0);
                _mm256_storeu_ps(sel[1][x]+y, mask_1);
                _mm256_storeu_ps(sel[2][x]+y, mask_2);                                
        }
    }

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r.exr", &sel[0], img_width, img_height);
        write_channel_exr("temp/sel_g.exr", &sel[1], img_width, img_height);
        write_channel_exr("temp/sel_b.exr", &sel[2], img_width, img_height);
        cout << "\t - Selection Maps done" << endl;
    }

   // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = 5};
    buffer sel_filtered;
    allocate_buffer_zero(&sel_filtered, img_width, img_height);
    filtering_basic_f1_VEC(sel_filtered, sel, c, c_var, p_sel, img_width, img_height);

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r_filtered.exr", &sel_filtered[0], img_width, img_height);
        write_channel_exr("temp/sel_g_filtered.exr", &sel_filtered[1], img_width, img_height);
        write_channel_exr("temp/sel_b_filtered.exr", &sel_filtered[2], img_width, img_height);
        cout << "\t - Filter Selection Maps done" << endl;
    }

    // ----------------------------------------------
    // (7) Candidate Filter averaging
    // ----------------------------------------------
    scalar w_r, w_g, w_b, norm;
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < img_width; x++){
            for (int y = 0; y < img_height; y++){

                // Retrieve weights and normalization term => such that weights sum up to 1
                w_r = sel_filtered[0][x][y];
                w_g = sel_filtered[1][x][y];
                w_b = sel_filtered[2][x][y];

                norm = w_r + w_g + w_b;
            
                // Set candidate r as base => for boundary parts and pixels with norm == 0
                out_img[i][x][y] = r[i][x][y];

                // Averaging of candidate filters
                if (norm > EPSILON and norm != INFINITY){
                    out_img[i][x][y] =  ((w_r * r[i][x][y]) 
                                      + (w_g * g[i][x][y])
                                      + (w_b * b[i][x][y]))/norm;
                }
            
            }
        }
    }

    // DEBUG PART 
    if(DEBUG) {
        write_buffer_exr("temp/pass1.exr", &out_img, img_width, img_height);
        cout << "\t - Pass1 done" << endl;
    }
  

    // ----------------------------------------------
    // (8) Memory Deallocation
    // ----------------------------------------------
    // Free filtered filters
    free_buffer(&f_filtered, img_width);
    free_buffer(&f_var_filtered, img_width);

    // Free candidate filters and their derivates
    free_buffer(&r, img_width);
    free_buffer(&g, img_width);
    free_buffer(&b, img_width);

    // Free sure estimates (unfiltered and filtered)
    free_buffer(&sure, img_width);
    free_buffer(&e, img_width);
    
    // Free selection maps (unfiltered and filtered)
    free_buffer(&sel, img_width);
    free_buffer(&sel_filtered, img_width);


 }
