#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
#include "../flt_restructure.hpp"
#include "../memory_mgmt.hpp"

#include "../denoise.h"
#include <immintrin.h>
// #include "../avx_mathfun.h"

#define BLOCK_SIZE 64

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
 void basic_restructure5(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height){

    if(DEBUG) {
        cout << "--------------------------------------------------" << endl;
        cout << " Starting Algorithm " << endl;
    }

    // ----------------------------------------------
    // (1) Buffer allocations
    // ----------------------------------------------
    Flt_parameters p_pre = { .kc = 1., .kf = INFINITY, .tau = 0., .f = 3, .r = 5};
    buffer f_filtered;
    buffer f_var_filtered;
    allocate_buffer_zero(&f_filtered, img_width, img_height);
    allocate_buffer_zero(&f_var_filtered, img_width, img_height);

    Flt_parameters p_all[3];
    p_all[0] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    p_all[1] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    p_all[2] = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};   // Fixed Variable: kc=INF => is exploited in filtering
    buffer r, g, b;
    allocate_buffer_zero(&r, img_width, img_height);
    allocate_buffer_zero(&g, img_width, img_height);
    allocate_buffer_zero(&b, img_width, img_height);

    buffer sure;
    allocate_buffer(&sure, img_width, img_height); // Zero allocation done in Sure

    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    buffer e;
    allocate_buffer_zero(&e, img_width, img_height);

    buffer sel;
    allocate_buffer(&sel, img_width, img_height);

    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = 5};
    buffer sel_filtered;
    allocate_buffer_zero(&sel_filtered, img_width, img_height);
    

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    
    feature_prefiltering_BLK(f_filtered, f_var_filtered, f, f_var, p_pre, img_width, img_height);
    

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------

    // (a) Candidate Filters
    for(int X0 = R+3; X0 < img_width - R - 4 - BLOCK_SIZE; X0 += BLOCK_SIZE) {
        for(int Y0 = R+3; Y0 < img_height - R - 4 - BLOCK_SIZE; Y0 += BLOCK_SIZE) {
            std::cout << "Filtering " << X0 << " " << Y0 << std::endl;
            candidate_filtering_all_BLK(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, X0, Y0, BLOCK_SIZE, BLOCK_SIZE);
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to F_R = F_B)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < 0+img_width; xp++){
            for(int yp = 0; yp < 0+R + 3; yp++){
                r[i][xp][yp] = c[i][xp][yp];
                r[i][xp][img_height - yp - 1] = c[i][xp][img_height - yp - 1];
                b[i][xp][yp] = c[i][xp][yp];
                b[i][xp][img_height - yp - 1] = c[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < 0+R + 3; xp++){
            for (int yp = 0+R + 3 ; yp < 0+img_height - R - 3; yp++){
            
                r[i][xp][yp] = c[i][xp][yp];
                r[i][img_width - xp - 1][yp] = c[i][img_width - xp - 1][yp];
                b[i][xp][yp] = c[i][xp][yp];
                b[i][img_width - xp - 1][yp] = c[i][img_width - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since F_G != F_R
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < 0+img_width; xp++){
            for(int yp = 0; yp < 0+R + 3; yp++){
                g[i][xp][yp] = c[i][xp][yp];
                g[i][xp][img_height - yp - 1] = c[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < 0+R + 3; xp++){
            for (int yp = 0+R + 3 ; yp < 0+img_height - R - 3; yp++){
                g[i][xp][yp] = c[i][xp][yp];
                g[i][img_width - xp - 1][yp] = c[i][img_width - xp - 1][yp];
            }
        }
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    sure_all_BLK(sure, c, c_var, r, g, b, img_width, img_height);

    // (b) Filter error estimates
    filtering_basic_f1_BLK(e, sure, c, c_var, p_sure, img_width, img_height);

     // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    
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

   // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    filtering_basic_f1_BLK(sel_filtered, sel, c, c_var, p_sel, img_width, img_height);

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
