#include <iostream>
#include <cmath>
#include "../exr.h"
#include "../flt.hpp"
#include "../flt_restructure.hpp"
#include "../memory_mgmt.hpp"

#include "../denoise.h"


using namespace std;

/*! -------------------------------------------------------
	Restructured Implementaton without Precomputation
    
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
 void basic_restructure(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height){

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
    buffer f_filtered, f_var_filtered;
    allocate_buffer_zero(&f_filtered, img_width, img_height);
    allocate_buffer_zero(&f_var_filtered, img_width, img_height);
    filtering_basic(f_filtered, f, f, f_var, p_pre, img_width, img_height);
    filtering_basic(f_var_filtered, f_var, f, f_var, p_pre, img_width, img_height);
    
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
    Flt_parameters p_r = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 1, .r = R};
    buffer r;
    allocate_buffer_zero(&r, img_width, img_height);
    candidate_filtering(r, c, c_var, f_filtered, f_var_filtered, p_r, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_FIRST.exr", &r, img_width, img_height);
        cout << "\t - Candidate FIRST done" << endl;
    }

    // (b) Candidate Filter: SECOND
    Flt_parameters p_g = { .kc = 2.0, .kf = 0.6, .tau = 0.001, .f = 3, .r = R};
    buffer g;
    allocate_buffer_zero(&g, img_width, img_height);
    candidate_filtering(g, c, c_var, f_filtered, f_var_filtered, p_g, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_SECOND.exr", &g, img_width, img_height);
        cout << "\t - Candidate SECOND done" << endl;
    }

    // (c) Candidate Filter: THIRD
    Flt_parameters p_b = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r = R};
    buffer b;
    allocate_buffer_zero(&b, img_width, img_height);
    candidate_filtering_THIRD(b, c, c_var, f_filtered, f_var_filtered, p_b, img_width, img_height);
    
    // DEBUGGING PART
    if(DEBUG) {
        write_buffer_exr("temp/candidate_THIRD.exr", &b, img_width, img_height);
        cout << "\t - Candidate THIRD done" << endl;
    }

    // ----------------------------------------------
    // (4) Filtering SURE error estimates
    // ----------------------------------------------

    // (a) Compute SURE error estimates
    buffer sure;
    allocate_buffer_zero(&sure, img_width, img_height);
    sure_all(sure, c, c_var, r, g, b, img_width, img_height);
    
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
    filtering_basic(e, sure, c, c_var, p_sure, img_width, img_height);
    
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
    
    // Compute selection maps
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){
                sel[0][x][y] = e[0][x][y] < e[1][x][y] && e[0][x][y] < e[2][x][y];
                sel[1][x][y] = e[1][x][y] < e[0][x][y] && e[1][x][y] < e[2][x][y];
                sel[2][x][y] = e[2][x][y] < e[0][x][y] && e[1][x][y] < e[2][x][y];
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
    filtering_basic(sel_filtered, sel, c, c_var, p_sel, img_width, img_height);

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
