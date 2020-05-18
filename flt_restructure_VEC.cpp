#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"
#include <immintrin.h>


void sure_all_VEC(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    __m256 d_r_vec, d_g_vec, d_b_vec, v_vec, c_vec;
    

    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y+=8){
            __m256 sure_r_vec = _mm256_setzero_ps();
            __m256 sure_g_vec = _mm256_setzero_ps();
            __m256 sure_b_vec = _mm256_setzero_ps();
                
            c_vec = _mm256_loadu_ps(c[0][x]+y);
            d_r_vec = _mm256_loadu_ps(cand_r[0][x]+y);
            d_g_vec = _mm256_loadu_ps(cand_g[0][x]+y);
            d_b_vec = _mm256_loadu_ps(cand_b[0][x]+y);
            v_vec = _mm256_loadu_ps(c_var[0][x]+y);

            // d_r = d_r - c
            d_r_vec = _mm256_sub_ps(d_r_vec, c_vec);
            d_g_vec = _mm256_sub_ps(d_g_vec, c_vec);
            d_b_vec = _mm256_sub_ps(d_b_vec, c_vec);

            // Squared
            v_vec = _mm256_mul_ps(v_vec, v_vec);
            d_r_vec = _mm256_mul_ps(d_r_vec, d_r_vec);
            d_g_vec = _mm256_mul_ps(d_g_vec, d_g_vec);
            d_b_vec = _mm256_mul_ps(d_b_vec, d_b_vec);

            // Difference d_r = d_r-v
            d_r_vec = _mm256_sub_ps(d_r_vec, v_vec);
            d_g_vec = _mm256_sub_ps(d_g_vec, v_vec);
            d_b_vec = _mm256_sub_ps(d_b_vec, v_vec);

            // Summing up
            sure_r_vec = _mm256_add_ps(sure_r_vec, d_r_vec);
            sure_g_vec = _mm256_add_ps(sure_g_vec, d_g_vec);
            sure_b_vec = _mm256_add_ps(sure_b_vec, d_b_vec);
                
            _mm256_storeu_ps(sure[0][x]+y, sure_r_vec);
            _mm256_storeu_ps(sure[1][x]+y, sure_g_vec);
            _mm256_storeu_ps(sure[2][x]+y, sure_b_vec);
        }
    }



    for (int i = 1; i < 3; i++){ 
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y+=8){
                __m256 sure_r_vec = _mm256_loadu_ps(sure[0][x]+y);
                __m256 sure_g_vec = _mm256_loadu_ps(sure[1][x]+y);
                __m256 sure_b_vec = _mm256_loadu_ps(sure[2][x]+y);
                
                c_vec = _mm256_loadu_ps(c[i][x]+y);
                d_r_vec = _mm256_loadu_ps(cand_r[i][x]+y);
                d_g_vec = _mm256_loadu_ps(cand_g[i][x]+y);
                d_b_vec = _mm256_loadu_ps(cand_b[i][x]+y);
                v_vec = _mm256_loadu_ps(c_var[i][x]+y);

                // d_r = d_r - c
                d_r_vec = _mm256_sub_ps(d_r_vec, c_vec);
                d_g_vec = _mm256_sub_ps(d_g_vec, c_vec);
                d_b_vec = _mm256_sub_ps(d_b_vec, c_vec);

                // Squared
                v_vec = _mm256_mul_ps(v_vec, v_vec);
                d_r_vec = _mm256_mul_ps(d_r_vec, d_r_vec);
                d_g_vec = _mm256_mul_ps(d_g_vec, d_g_vec);
                d_b_vec = _mm256_mul_ps(d_b_vec, d_b_vec);

                // Difference d_r = d_r-v
                d_r_vec = _mm256_sub_ps(d_r_vec, v_vec);
                d_g_vec = _mm256_sub_ps(d_g_vec, v_vec);
                d_b_vec = _mm256_sub_ps(d_b_vec, v_vec);

                // Summing up
                sure_r_vec = _mm256_add_ps(sure_r_vec, d_r_vec);
                sure_g_vec = _mm256_add_ps(sure_g_vec, d_g_vec);
                sure_b_vec = _mm256_add_ps(sure_b_vec, d_b_vec);
                
                _mm256_storeu_ps(sure[0][x]+y, sure_r_vec);
                _mm256_storeu_ps(sure[1][x]+y, sure_g_vec);
                _mm256_storeu_ps(sure[2][x]+y, sure_b_vec);
            }
        }
    }
}