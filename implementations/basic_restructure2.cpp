#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
#include "../flt_restructure.hpp"
#include "../memory_mgmt.hpp"

#include "../denoise.h"


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
        - W (int)       Image Width
        - H (int)      Image Height
å
    \return void --> denoised image in buffer out_img
 */
 void basic_restructure2(scalar* out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H){

    if(DEBUG) {
        cout << "--------------------------------------------------" << endl;
        cout << " Starting Algorithm " << endl;
    }
     
    int WH = W*H;

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    Flt_parameters p_pre = { .kc = 1., .kf = INFINITY, .tau = 0., .f = 3, .r = 5};
    scalar* f_filtered;
    scalar* f_var_filtered;
    f_filtered = (scalar*) calloc(3 * WH, sizeof(scalar));
    f_var_filtered = (scalar*) calloc(3 * WH, sizeof(scalar));

    feature_prefiltering(f_filtered, f_var_filtered, f, f_var, p_pre, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/albedo_filtered_.exr", f_filtered, W, H);
        write_channel_exr("temp/depth_filtered_.exr",  f_filtered + WH, W, H);
        write_channel_exr("temp/normal_filtered_.exr", f_filtered + 2*WH, W, H);
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
    scalar* r;
    scalar* g;
    scalar* b;
    r = (scalar*) calloc(3 * WH, sizeof(scalar)); 
    g = (scalar*) calloc(3 * WH, sizeof(scalar)); 
    b = (scalar*) calloc(3 * WH, sizeof(scalar));       
    candidate_filtering_all(r, g, b, c, c_var, f_filtered, f_var_filtered, p_all, W, H);

    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST_.exr", r, W, H);
        write_buffer_exr("temp/candidate_SECOND_.exr", g, W, H);
        write_buffer_exr("temp/candidate_THIRD_.exr", b, W, H);
        cout << "\t - Candidates done" << endl;
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    scalar* sure;
    sure = (scalar*) calloc(3 * W * H, sizeof(scalar)); 
    sure_all(sure, c, c_var, r, g, b, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/sure_r_.exr", sure, W, H);
        write_channel_exr("temp/sure_g_.exr", sure + WH, W, H);
        write_channel_exr("temp/sure_b_.exr", sure + 2*WH, W, H);   
        cout << "\t - Sure Error Estimates done" << endl;
    }

    // (b) Filter error estimates
    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    scalar* e;
    e = (scalar*) calloc(3 * W * H,  sizeof(scalar)); 
    filtering_basic(e, sure, c, c_var, p_sure, W, H);
    
    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/e_r_.exr", e, W, H);
        write_channel_exr("temp/e_g_.exr", e + WH, W, H);
        write_channel_exr("temp/e_b_.exr", e + 2*WH, W, H);
        cout << "\t - Filtered Sure Error Estimates done" << endl;
    }

     // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    scalar* sel;
    sel = (scalar*) calloc(3 * W * H, sizeof(scalar));  
    
    // Compute selection maps
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){
                sel[0 * WH + x * W + y] = e[0 * WH + x * W + y] < e[1 * WH + x * W + y] && e[0 * WH + x * W + y] < e[2 * WH + x * W + y];
                sel[1 * WH + x * W + y] = e[1 * WH + x * W + y] < e[0 * WH + x * W + y] && e[1 * WH + x * W + y] < e[2 * WH + x * W + y];
                sel[2 * WH + x * W + y] = e[2 * WH + x * W + y] < e[0 * WH + x * W + y] && e[1 * WH + x * W + y] < e[2 * WH + x * W + y];
        }
    }

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r_.exr", sel, W, H);
        write_channel_exr("temp/sel_g_.exr", sel + WH, W, H);
        write_channel_exr("temp/sel_b_.exr", sel + 2*WH, W, H);
        cout << "\t - Selection Maps done" << endl;
    }

   // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 5};
    scalar* sel_filtered;
    sel_filtered = (scalar*) calloc(3 * W * H, sizeof(scalar)); 
    filtering_basic(sel_filtered, sel, c, c_var, p_sel, W, H);

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r_filtered_.exr", sel_filtered , W, H);
        write_channel_exr("temp/sel_g_filtered_.exr", sel_filtered + WH, W, H);
        write_channel_exr("temp/sel_b_filtered_.exr", sel_filtered + 2*WH, W, H);
        cout << "\t - Filter Selection Maps done" << endl;
    }

    // ----------------------------------------------
    // (7) Candidate Filter averaging
    // ----------------------------------------------
    scalar w_r, w_g, w_b, norm;
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y++){

                // Retrieve weights and normalization term => such that weights sum up to 1
                w_r = sel_filtered[0 * WH + x * W + y];
                w_g = sel_filtered[1 * WH + x * W + y];
                w_b = sel_filtered[2 * WH + x * W + y];

                norm = w_r + w_g + w_b;
            
                // Set candidate r as base => for boundary parts and pixels with norm == 0
                out_img[i * WH + x * W + y] = r[i * WH + x * W + y];

                // Averaging of candidate filters
                if (norm > EPSILON and norm != INFINITY){
                    out_img[i * WH + x * W + y] =  ((w_r * r[i * WH + x * W + y]) 
                                                  + (w_g * g[i * WH + x * W + y])
                                                  + (w_b * b[i * WH + x * W + y]))/norm;
                }
            
            }
        }
    }

    // DEBUG PART 
    if(DEBUG) {
        write_buffer_exr("temp/pass1_.exr", out_img, W, H);
        cout << "\t - Pass1 done" << endl;
    }
  

    // ----------------------------------------------
    // (8) Memory Deallocation
    // ----------------------------------------------

    // Free filtered filters
    free(f_filtered);
    free(f_var_filtered);

    // Free candidate filters and their derivates
    free(r);
    free(g);
    free(b);

    // Free sure estimates (unfiltered and filtered)
    free(sure);
    free(e);
    
    // Free selection maps (unfiltered and filtered)
    free(sel);
    free(sel_filtered);


 }
