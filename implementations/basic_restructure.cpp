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
 void basic_restructure1(scalar* out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H){

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

    filtering_basic(f_filtered, f, f, f_var, p_pre, W, H); 
    filtering_basic(f_var_filtered, f_var, f, f_var, p_pre, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/albedo_filtered.exr", f_filtered, W, H);
        write_channel_exr("temp/depth_filtered.exr",  f_filtered + WH, W, H);
        write_channel_exr("temp/normal_filtered.exr", f_filtered + 2*WH, W, H);
        cout << "\t - Feature Prefiltering done" << endl;
    }

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------
    // (a) Candidate Filter: FIRST
    Flt_parameters p_all[3];
    p_all[0] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    scalar* r;
    r = (scalar*) calloc(3 * WH, sizeof(scalar));     

    candidate_filtering(r, c, c_var, f_filtered, f_var_filtered, p_all[0], W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST.exr", r, W, H);
        cout << "\t - Candidate FIRST done" << endl;
    }

    // (b) Candidate Filter: SECOND
    p_all[1] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    scalar* g;
    g = (scalar*) calloc(3 * WH, sizeof(scalar));   
    candidate_filtering(g, c, c_var, f_filtered, f_var_filtered, p_all[1], W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_SECOND.exr", g, W, H);
        cout << "\t - Candidate SECOND done" << endl;
    }

    // (c) Candidate Filter: THIRD
     p_all[2] = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};   // Fixed Variable: kc=INF => is exploited in filtering
    scalar* b;
    b = (scalar*) calloc(3 * WH, sizeof(scalar)); 
    candidate_filtering(b, c, c_var, f_filtered, f_var_filtered, p_all[2], W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_THIRD.exr", b, W, H);
        cout << "\t - Candidate THIRD done" << endl;
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
        write_channel_exr("temp/sure_r.exr", sure, W, H);
        write_channel_exr("temp/sure_g.exr", sure + WH, W, H);
        write_channel_exr("temp/sure_b.exr", sure + 2*WH, W, H);   
        cout << "\t - Sure Error Estimates done" << endl;
    }

    // (b) Filter error estimates
    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    scalar* e;
    e = (scalar*) calloc(3 * W * H, sizeof(scalar)); 
    filtering_basic(e, sure, c, c_var, p_sure, W, H);
    
    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/e_r.exr", e, W, H);
        write_channel_exr("temp/e_g.exr", e + WH, W, H);
        write_channel_exr("temp/e_b.exr", e + 2*WH, W, H);
        cout << "\t - Filtered Sure Error Estimates done" << endl;
    }

    // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    scalar* sel;
    sel = (scalar*) (scalar*) calloc(3 * W * H, sizeof(scalar));  
    
    // Compute selection maps
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){
                sel[0 + 3 *  (x * W + y)] = e[0 + 3 *  (x * W + y)] < e[1 + 3 *  (x * W + y)] && e[0 + 3 *  (x * W + y)] < e[2 + 3 *  (x * W + y)];
                sel[1 + 3 *  (x * W + y)] = e[1 + 3 *  (x * W + y)] < e[0 + 3 *  (x * W + y)] && e[1 + 3 *  (x * W + y)] < e[2 + 3 *  (x * W + y)];
                sel[2 + 3 *  (x * W + y)] = e[2 + 3 *  (x * W + y)] < e[0 + 3 *  (x * W + y)] && e[1 + 3 *  (x * W + y)] < e[2 + 3 *  (x * W + y)];
        }
    }

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r.exr", sel, W, H);
        write_channel_exr("temp/sel_g.exr", sel + WH, W, H);
        write_channel_exr("temp/sel_b.exr", sel + 2*WH, W, H);
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
        write_channel_exr("temp/sel_r_filtered.exr", sel_filtered , W, H);
        write_channel_exr("temp/sel_g_filtered.exr", sel_filtered + WH, W, H);
        write_channel_exr("temp/sel_b_filtered.exr", sel_filtered + 2*WH, W, H);
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
                w_r = sel_filtered[0 + 3 *  (x * W + y)];
                w_g = sel_filtered[1 + 3 *  (x * W + y)];
                w_b = sel_filtered[2 + 3 *  (x * W + y)];

                norm = w_r + w_g + w_b;
            
                // Set candidate r as base => for boundary parts and pixels with norm == 0
                out_img[i + 3 *  (x * W + y)] = r[i + 3 *  (x * W + y)];

                // Averaging of candidate filters
                if (norm > EPSILON and norm != INFINITY){
                    out_img[i + 3 *  (x * W + y)] =  ((w_r * r[i + 3 *  (x * W + y)]) 
                                                  + (w_g * g[i + 3 *  (x * W + y)])
                                                  + (w_b * b[i + 3 *  (x * W + y)]))/norm;
                }
            }
        }
    }

    // DEBUG PART 
    if(DEBUG) {
        write_buffer_exr("temp/pass1.exr", out_img, W, H);
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
