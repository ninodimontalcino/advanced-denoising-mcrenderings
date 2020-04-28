#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
#include "../fltopcount.hpp"
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
        - img_width (int)       Image Width
        - img_height (int)      Image Height
å
    \return void --> denoised image in buffer out_img
 */
 void optopcount(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height){

    if(DEBUG) {
        cout << "--------------------------------------------------" << endl;
        cout << " Starting Algorithm " << endl;
    }

    Flt_parameters all_params[6];
    all_params[0] = { .kc = 1., .kf = INFINITY, .tau = 0., .f = 3, .r = 5}; // Prefiltering
    all_params[1] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R}; // candidate FIRST
    all_params[2] = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R}; // candidate SECOND
    all_params[3] = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R}; // candidate THIRD
    all_params[4] = { .kc = 1.0, .kf = INFINITY, .tau = 0.001, .f = 1, .r = 1}; // filter error estimate
    all_params[5] = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = 5}; // filter selection map

    // ----------------------------------------------
    // (1) Sample Variance Scaling
    // ----------------------------------------------
    // !!! => not necessary in our case since sample variance is already computed in renderer (only 1 buffer output due to uniform random MC sampling)


    // ----------------------------------------------
    // Weights precomputation for prefiltering
    // ----------------------------------------------

    bufferweightset weights_pref, weights;
    scalar weights_sums[6];
    allocate_buffer_weights(&weights_pref, img_width, img_height, 1);
    precompute_colors_pref(weights_pref, weights_sums, f, f_var, img_width, img_height, all_params[0]);
    if(DEBUG)
        cout << "\t - Precomputation of prefiltering weights done" << endl;

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    Flt_parameters p_pre = all_params[0];
    buffer f_filtered, f_var_filtered;
    allocate_buffer(&f_filtered, img_width, img_height);
    allocate_buffer(&f_var_filtered, img_width, img_height);
    flt_buffer_opcount(f_filtered, f, f, f_var, all_params, 0, img_width, img_height, weights_pref);
    flt_buffer_opcount(f_var_filtered, f_var, f, f_var, all_params, 0, img_width, img_height, weights_pref);
    
    free_buffer_weights(&weights_pref, img_width, img_height, 1);

    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/albedo_filtered.exr", &f_filtered[0], img_width, img_height);
        write_channel_exr("temp/depth_filtered.exr", &f_filtered[1], img_width, img_height);
        write_channel_exr("temp/normal_filtered.exr", &f_filtered[2], img_width, img_height);
        cout << "\t - Feature Prefiltering done" << endl;
    }
    
    // ----------------------------------------------
    // Weights precomputation for other stages
    // ----------------------------------------------
    allocate_buffer_weights(&weights, img_width, img_height, 2); // need 5 but my laptop freezes 
    precompute_weights(weights, weights_sums, c, c_var, f_filtered, f_var_filtered, img_width, img_height, all_params);

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------
    // (a) Candidate Filter: FIRST
    Flt_parameters p_r = all_params[1];
    buffer r, d_r;
    allocate_buffer(&r, img_width, img_height);
    allocate_buffer(&d_r, img_width, img_height);
    flt(r, d_r, c, c, c_var, f_filtered, f_var_filtered, p_r, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST.exr", &r, img_width, img_height);
        cout << "\t - Candidate FIRST done" << endl;
    }

    // (b) Candidate Filter: SECOND
    Flt_parameters p_g = all_params[2];
    buffer g, d_g;
    allocate_buffer(&g, img_width, img_height);
    allocate_buffer(&d_g, img_width, img_height);
    flt(g, d_g, c, c, c_var, f_filtered, f_var_filtered, p_g, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_SECOND.exr", &g, img_width, img_height);
        cout << "\t - Candidate SECOND done" << endl;
    }

    // (c) Candidate Filter: THIRD
    Flt_parameters p_b = all_params[3];
    buffer b, d_b;
    allocate_buffer(&b, img_width, img_height);
    allocate_buffer(&d_b, img_width, img_height);
    flt(b, d_b, c, c, c_var, f_filtered, f_var_filtered, p_b, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_THIRD.exr", &b, img_width, img_height);
        cout << "\t - Candidate THIRD done" << endl;
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    channel sure_r, sure_g, sure_b;
    allocate_channel(&sure_r, img_width, img_height);
    allocate_channel(&sure_g, img_width, img_height);
    allocate_channel(&sure_b, img_width, img_height);
    sure(sure_r, c, c_var, r, d_r, img_width, img_height);
    sure(sure_g, c, c_var, g, d_g, img_width, img_height);
    sure(sure_b, c, c_var, b, d_b, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_channel_exr("temp/sure_r.exr", &sure_r, img_width, img_height);
        write_channel_exr("temp/sure_g.exr", &sure_g, img_width, img_height);
        write_channel_exr("temp/sure_b.exr", &sure_b, img_width, img_height);   
        cout << "\t - Sure Error Estimates done" << endl;
    }

    // (b) Filter error estimates
    Flt_parameters p_sure = all_params[4];
    channel e_r, e_g, e_b;
    allocate_channel(&e_r, img_width, img_height);
    allocate_channel(&e_g, img_width, img_height);
    allocate_channel(&e_b, img_width, img_height);
    flt_channel_basic(e_r, sure_r, c, c_var, p_sure, img_width, img_height);
    flt_channel_basic(e_g, sure_g, c, c_var, p_sure, img_width, img_height);
    flt_channel_basic(e_b, sure_b, c, c_var, p_sure, img_width, img_height);
    
    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/e_r.exr", &e_r, img_width, img_height);
        write_channel_exr("temp/e_g.exr", &e_g, img_width, img_height);
        write_channel_exr("temp/e_b.exr", &e_b, img_width, img_height);
        cout << "\t - Filtered Sure Error Estimates done" << endl;
    }

    // ----------------------------------------------
    // (5) Compute Binary Selection Maps
    // ----------------------------------------------
    channel sel_r, sel_g, sel_b;
    allocate_channel(&sel_r, img_width, img_height);
    allocate_channel(&sel_g, img_width, img_height);
    allocate_channel(&sel_b, img_width, img_width);
    
    // Compute selection maps
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){
                sel_r[x][y] = e_r[x][y] < e_g[x][y] && e_r[x][y] < e_b[x][y]; // && d_r[0][x][y] < d_g[0][x][y];
                sel_g[x][y] = e_g[x][y] < e_r[x][y] && e_g[x][y] < e_b[x][y];
                sel_b[x][y] = e_b[x][y] < e_r[x][y] && e_g[x][y] < e_b[x][y];
        }
    }

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r.exr", &sel_r, img_width, img_height);
        write_channel_exr("temp/sel_g.exr", &sel_g, img_width, img_height);
        write_channel_exr("temp/sel_b.exr", &sel_b, img_width, img_height);
        cout << "\t - Selection Maps done" << endl;
    }

    // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    Flt_parameters p_sel = all_params[5];
    channel sel_r_filtered, sel_g_filtered,  sel_b_filtered;
    allocate_channel(&sel_r_filtered, img_width, img_height);
    allocate_channel(&sel_g_filtered, img_width, img_height);
    allocate_channel(&sel_b_filtered, img_width, img_height);
    flt_channel_basic(sel_r_filtered, sel_r, c, c_var, p_sel, img_width, img_height);
    flt_channel_basic(sel_g_filtered, sel_g, c, c_var, p_sel, img_width, img_height);
    flt_channel_basic(sel_b_filtered, sel_b, c, c_var, p_sel, img_width, img_height);

    // DEBUG PART
    if(DEBUG) {
        write_channel_exr("temp/sel_r_filtered.exr", &sel_r_filtered, img_width, img_height);
        write_channel_exr("temp/sel_g_filtered.exr", &sel_g_filtered, img_width, img_height);
        write_channel_exr("temp/sel_b_filtered.exr", &sel_b_filtered, img_width, img_height);
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
                w_r = sel_r_filtered[x][y];
                w_g = sel_g_filtered[x][y];
                w_b = sel_b_filtered[x][y];

                norm = w_r + w_g + w_b;

                // Set candidate r as base => for boundary parts and pixels with norm == 0
                out_img[i][x][y] = r[i][x][y];

                // Averaging of candidate filters
                if (norm > EPSILON and norm != INFINITY){
                    out_img[i][x][y] =  (w_r * r[i][x][y] / norm) 
                                      + (w_g * g[i][x][y] / norm)
                                      + (w_b * b[i][x][y] / norm);
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
    // (8) Second Pass-Filtering
    // ----------------------------------------------
    // => not necessary since we use a one buffer approach (due to independet MC samples in the renderer) at the moment
    // Therefore we can use pass1 as output
    
    // Flt_parameters p_final= { .kc = 0.45, .kf = INFINITY, .tau = 0.0001, .f = 1, .r = R};
    // flt_buffer_basic(out_img, pass1, pass1, c_var, p_final, img_width, img_height);    

    // ----------------------------------------------
    // (9) Memory Deallocation
    // ----------------------------------------------
    // Free filtered filters
    free_buffer(&f_filtered, img_width);
    free_buffer(&f_var_filtered, img_width);

    // Free candidate filters and their derivates
    free_buffer(&r, img_width);
    free_buffer(&d_r, img_width);
    free_buffer(&g, img_width);
    free_buffer(&d_g, img_width);
    free_buffer(&b, img_width);
    free_buffer(&d_b, img_width);

    // Free sure estimates (unfiltered and filtered)
    free_channel(&sure_r, img_width);
    free_channel(&sure_g, img_width);
    free_channel(&sure_b, img_width);
    free_channel(&e_r, img_width);
    free_channel(&e_g, img_width);
    free_channel(&e_b, img_width);

    // Free selection maps (unfiltered and filtered)
    free_channel(&sel_r, img_width);
    free_channel(&sel_g, img_width);
    free_channel(&sel_b, img_width);
    free_channel(&sel_r_filtered, img_width);
    free_channel(&sel_g_filtered, img_width);
    free_channel(&sel_b_filtered, img_width);


 }
