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
	Restructured Implementaton with Precomputation (VECTORIZED) 
    => Introduction of additional memory structure for overlapping computations
    
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
        - W (int)       Image Width
        - H (int)      Image Height
å
    \return void --> denoised image in buffer out_img
 */
 void basic_restructure6(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int W, int H){

    int WH = W*H;

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
    allocate_buffer_zero(&f_filtered, W, H);
    allocate_buffer_zero(&f_var_filtered, W, H);

    Flt_parameters p_all[3];
    p_all[0] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    p_all[1] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    p_all[2] = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};   // Fixed Variable: kc=INF => is exploited in filtering
    buffer r, g, b;
    allocate_buffer_zero(&r, W, H);
    allocate_buffer_zero(&g, W, H);
    allocate_buffer_zero(&b, W, H);

    buffer sure;
    allocate_buffer(&sure, W, H); // Zero allocation done in Sure

    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    buffer e;
    allocate_buffer_zero(&e, W, H);

    buffer sel;
    allocate_buffer(&sel, W, H);

    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = 5};
    buffer sel_filtered;
    allocate_buffer_zero(&sel_filtered, W, H);
    

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    
    feature_prefiltering_VEC(f_filtered, f_var_filtered, f, f_var, p_pre, W, H);
    

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------


    // (A) GRADIENT COMPUTATION
    // ---------------------------------
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));
    int F_MIN = 1;

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F_MIN; x < W - R - F_MIN; ++x) {
            for(int y =  R+F_MIN; y < H - R - F_MIN; y+=8) {
                
                // (1) Loading
                features_vec  = _mm256_loadu_ps(f[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(f[i][x-1] + y);
                diffR_sqr_vec = _mm256_loadu_ps(f[i][x+1]+ y);
                diffU_sqr_vec = _mm256_loadu_ps(f[i][x] + y-1);
                diffD_sqr_vec = _mm256_loadu_ps(f[i][x] + y+1);

                // (2) Computing Squared Differences
                diffL_sqr_vec = _mm256_sub_ps(features_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_sub_ps(features_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_sub_ps(features_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_sub_ps(features_vec, diffD_sqr_vec);

                diffL_sqr_vec = _mm256_mul_ps(diffL_sqr_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_mul_ps(diffR_sqr_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_mul_ps(diffU_sqr_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_mul_ps(diffD_sqr_vec, diffD_sqr_vec);

                // (3) Final Gradient Computation
                tmp_1 = _mm256_min_ps(diffL_sqr_vec, diffR_sqr_vec);
                tmp_2 = _mm256_min_ps(diffU_sqr_vec, diffD_sqr_vec);

                tmp_1 = _mm256_add_ps(tmp_1, tmp_2);

                _mm256_storeu_ps(gradients+i * WH + x * W + y, tmp_1);

            } 
        }
    }

    // (B) GLOBAL MEMORY ALLOCATION
    // ---------------------------------
    int NEIGH_SIZE = (2*R + 1) * (2*R + 1);

    // (a) Feature Weights
    scalar* features_weights_r;
    scalar* features_weights_b;
    features_weights_r = (scalar*) malloc(NEIGH_SIZE * W * H * sizeof(scalar));
    features_weights_b = (scalar*) malloc(NEIGH_SIZE * W * H * sizeof(scalar));
    memset(features_weights_r, 0, NEIGH_SIZE*W*H*sizeof(scalar));
    memset(features_weights_b, 0, NEIGH_SIZE*W*H*sizeof(scalar));

    // (b) Temp Arrays for Convolution
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    temp = (scalar*) malloc(NEIGH_SIZE * W * H * sizeof(scalar));
    temp2_r = (scalar*) malloc(NEIGH_SIZE * W * H * sizeof(scalar));
    temp2_g = (scalar*) malloc(NEIGH_SIZE * W * H * sizeof(scalar));
    memset(temp, 0, NEIGH_SIZE*W*H*sizeof(scalar));


    // (..) MAIN FILTERING
    // ---------------------------------
    candidate_filtering_all_VEC_BLK_PREP(r, g, b, c, c_var, f_filtered, f_var_filtered, gradients, features_weights_r, features_weights_b, temp, temp2_r, temp2_g, p_all, W, H);

    // (a) Candidate Filters
    /*
    for(int X0 = R+3; X0 < W - R - 4 - BLOCK_SIZE; X0 += BLOCK_SIZE) {
        for(int Y0 = R+3; Y0 < H - R - 4 - BLOCK_SIZE; Y0 += BLOCK_SIZE) {
            candidate_filtering_all_BLK(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, X0, Y0, BLOCK_SIZE, BLOCK_SIZE);
        }
        candidate_filtering_all_BLK(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, X0, H - R - 4 - BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    }
    for(int Y0 = R+3; Y0 < H - R - 4 - BLOCK_SIZE; Y0 += BLOCK_SIZE)
        candidate_filtering_all_BLK(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, W - R - 4 - BLOCK_SIZE, Y0, BLOCK_SIZE, BLOCK_SIZE);
    */

    // (..) BORDER CASE HANDLING
    // ----------------------------------
    // Candidate FIRST and THIRD (due to F_R = F_B)
    int F_R = 1; 
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F_R; yp++){
                r[i][xp][yp] = c[i][xp][yp];
                r[i][xp][H - yp - 1] = c[i][xp][H - yp - 1];
                b[i][xp][yp] = c[i][xp][yp];
                b[i][xp][H - yp - 1] = c[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F_R; xp++){
            for (int yp = R + F_R ; yp < H - R - F_R; yp++){
            
                r[i][xp][yp] = c[i][xp][yp];
                r[i][W - xp - 1][yp] = c[i][W - xp - 1][yp];
                b[i][xp][yp] = c[i][xp][yp];
                b[i][W - xp - 1][yp] = c[i][W - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since F_G != F_R
    int F_G = 3; 
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F_G; yp++){
                g[i][xp][yp] = c[i][xp][yp];
                g[i][xp][H - yp - 1] = c[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F_G; xp++){
            for (int yp = R + F_G ; yp < H - R - F_G; yp++){
                g[i][xp][yp] = c[i][xp][yp];
                g[i][W - xp - 1][yp] = c[i][W - xp - 1][yp];
            }
        }
    }

    // (..) FREE MEMORY
    // ----------------------------------
    free(features_weights_r);
    free(features_weights_b);
    free(temp);

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    sure_all_VEC(sure, c, c_var, r, g, b, W, H);

    // (b) Filter error estimates
    filtering_basic_f1_VEC(e, sure, c, c_var, p_sure, W, H);

     // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    
    __m256 e0, e1, e2;
    __m256 mask0_0, mask1_0, mask_0;
    __m256 mask0_1, mask1_1, mask_1;
    __m256 mask0_2, mask1_2, mask_2;
    const __m256 ones = _mm256_set1_ps(1.);
    // Compute selection maps
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y+=8){

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
    filtering_basic_f1_VEC(sel_filtered, sel, c, c_var, p_sel, W, H);

    // ----------------------------------------------
    // (7) Candidate Filter averaging
    // ----------------------------------------------
    scalar w_r, w_g, w_b, norm;
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y++){

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
    free_buffer(&f_filtered, W);
    free_buffer(&f_var_filtered, W);

    // Free candidate filters and their derivates
    free_buffer(&r, W);
    free_buffer(&g, W);
    free_buffer(&b, W);

    // Free sure estimates (unfiltered and filtered)
    free_buffer(&sure, W);
    free_buffer(&e, W);
    
    // Free selection maps (unfiltered and filtered)
    free_buffer(&sel, W);
    free_buffer(&sel_filtered, W);


 }
