#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
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
 void basic_implementation(scalar *out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H){

    if(DEBUG) {
        cout << "--------------------------------------------------" << endl;
        cout << " Starting Algorithm " << endl;
    }

    int WH = W*H;

    // ----------------------------------------------
    // (1) Sample Variance Scaling
    // ----------------------------------------------
    // !!! => not necessary in our case since sample variance is already computed in renderer (only 1 buffer output due to uniform random MC sampling)

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    Flt_parameters p_pre = { .kc = 1., .kf = INFINITY, .tau = 0., .f = 3, .r = 5};
    scalar *f_filtered, *f_var_filtered;
    allocate_buffer(&f_filtered, W, H);
    allocate_buffer(&f_var_filtered, W, H);
    flt_buffer_basic(f_filtered, f, f, f_var, p_pre, W, H);
    flt_buffer_basic(f_var_filtered, f_var, f, f_var, p_pre, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/albedo_filtered.exr", f_filtered, W, H);
        write_channel_exr("temp/depth_filtered.exr", f_filtered + WH, W, H);
        write_channel_exr("temp/normal_filtered.exr", f_filtered + 2*WH, W, H);
        cout << "\t - Feature Prefiltering done" << endl;
    }

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------
    // (a) Candidate Filter: FIRST
    Flt_parameters p_r = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    scalar *r, *d_r;
    allocate_buffer(&r, W, H);
    allocate_buffer(&d_r, W, H);
    flt(r, d_r, c, c, c_var, f_filtered, f_var_filtered, p_r, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST.exr", r, W, H);
        cout << "\t - Candidate FIRST done" << endl;
    }

    // (b) Candidate Filter: SECOND
    Flt_parameters p_g = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    scalar *g, *d_g;
    allocate_buffer(&g, W, H);
    allocate_buffer(&d_g, W, H);
    flt(g, d_g, c, c, c_var, f_filtered, f_var_filtered, p_g, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_SECOND.exr", g, W, H);
        cout << "\t - Candidate SECOND done" << endl;
    }

    // (c) Candidate Filter: THIRD
    Flt_parameters p_b = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};
    scalar *b, *d_b;
    allocate_buffer(&b, W, H);
    allocate_buffer(&d_b, W, H);
    flt(b, d_b, c, c, c_var, f_filtered, f_var_filtered, p_b, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_THIRD.exr", b, W, H);
        cout << "\t - Candidate THIRD done" << endl;
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    scalar *sure_r, *sure_g, *sure_b;
    allocate_channel(&sure_r, W, H);
    allocate_channel(&sure_g, W, H);
    allocate_channel(&sure_b, W, H);
    sure(sure_r, c, c_var, r, d_r, W, H);
    sure(sure_g, c, c_var, g, d_g, W, H);
    sure(sure_b, c, c_var, b, d_b, W, H);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/sure_r_.exr", sure_r, W, H);
        write_channel_exr("temp/sure_g_.exr", sure_g, W, H);
        write_channel_exr("temp/sure_b_.exr", sure_b, W, H);   
        cout << "\t - Sure Error Estimates done" << endl;
    }

    // (b) Filter error estimates
    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1};
    scalar *e_r, *e_g, *e_b;
    allocate_channel(&e_r, W, H);
    allocate_channel(&e_g, W, H);
    allocate_channel(&e_b, W, H);
    flt_channel_basic(e_r, sure_r, c, c_var, p_sure, W, H);
    flt_channel_basic(e_g, sure_g, c, c_var, p_sure, W, H);
    flt_channel_basic(e_b, sure_b, c, c_var, p_sure, W, H);
    
    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/e_r.exr", e_r, W, H);
        write_channel_exr("temp/e_g.exr", e_g, W, H);
        write_channel_exr("temp/e_b.exr", e_b, W, H);
        cout << "\t - Filtered Sure Error Estimates done" << endl;
    }

    // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    scalar *sel_r, *sel_g, *sel_b;
    allocate_channel(&sel_r, W, H);
    allocate_channel(&sel_g, W, H);
    allocate_channel(&sel_b, W, W);
    
    // Compute selection maps
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){
                sel_r[x * W + y] = e_r[x * W + y] < e_g[x * W + y] && e_r[x * W + y] < e_b[x * W + y]; // && d_r[0][x][y] < d_g[0][x][y];
                sel_g[x * W + y] = e_g[x * W + y] < e_r[x * W + y] && e_g[x * W + y] < e_b[x * W + y];
                sel_b[x * W + y] = e_b[x * W + y] < e_r[x * W + y] && e_g[x * W + y] < e_b[x * W + y];
        }
    }

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r.exr", sel_r, W, H);
        write_channel_exr("temp/sel_g.exr", sel_g, W, H);
        write_channel_exr("temp/sel_b.exr", sel_b, W, H);
        cout << "\t - Selection Maps done" << endl;
    }

    // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 5};
    scalar *sel_r_filtered, *sel_g_filtered,  *sel_b_filtered;
    allocate_channel(&sel_r_filtered, W, H);
    allocate_channel(&sel_g_filtered, W, H);
    allocate_channel(&sel_b_filtered, W, H);
    flt_channel_basic(sel_r_filtered, sel_r, c, c_var, p_sel, W, H);
    flt_channel_basic(sel_g_filtered, sel_g, c, c_var, p_sel, W, H);
    flt_channel_basic(sel_b_filtered, sel_b, c, c_var, p_sel, W, H);

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r_filtered.exr", sel_r_filtered, W, H);
        write_channel_exr("temp/sel_g_filtered.exr", sel_g_filtered, W, H);
        write_channel_exr("temp/sel_b_filtered.exr", sel_b_filtered, W, H);
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
                w_r = sel_r_filtered[x * W + y];
                w_g = sel_g_filtered[x * W + y];
                w_b = sel_b_filtered[x * W + y];

                norm = w_r + w_g + w_b;

                // Set candidate r as base => for boundary parts and pixels with norm == 0
                out_img[i * WH + x * W + y] = r[i * WH + x * W + y];

                // Averaging of candidate filters
                if (norm > EPSILON and norm != INFINITY){
                    out_img[i * WH + x * W + y] =  (w_r * r[i * WH + x * W + y] / norm) 
                                      + (w_g * g[i * WH + x * W + y] / norm)
                                      + (w_b * b[i * WH + x * W + y] / norm);
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
    // (8) Second Pass-Filtering
    // ----------------------------------------------
    // => not necessary since we use a one buffer approach (due to independet MC samples in the renderer) at the moment
    // Therefore we can use pass1 as output
    
    // Flt_parameters p_final= { .kc = 0.45, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = R};
    // flt_buffer_basic(out_img, pass1, pass1, c_var, p_final, W, H);    

    // ----------------------------------------------
    // (9) Memory Deallocation
    // ----------------------------------------------
    // Free filtered filters
    free_buffer(&f_filtered);
    free_buffer(&f_var_filtered);

    // Free candidate filters and their derivates
    free_buffer(&r);
    free_buffer(&d_r);
    free_buffer(&g);
    free_buffer(&d_g);
    free_buffer(&b);
    free_buffer(&d_b);

    // Free sure estimates (unfiltered and filtered)
    free_channel(&sure_r);
    free_channel(&sure_g);
    free_channel(&sure_b);
    free_channel(&e_r);
    free_channel(&e_g);
    free_channel(&e_b);

    // Free selection maps (unfiltered and filtered)
    free_channel(&sel_r);
    free_channel(&sel_g);
    free_channel(&sel_b);
    free_channel(&sel_r_filtered);
    free_channel(&sel_g_filtered);
    free_channel(&sel_b_filtered);


 }
