#include <iostream>
#include <cmath>
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
        - img_width (int)       Image Width
        - img_height (int)      Image Height

    \return void --> denoised image in buffer out_img
 */
 void basic_implementation(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height){

     // ----------------------------------------------
    // (1) Sample Variance Scaling
    // ----------------------------------------------
    // !!! => not necessary in our case since sample variance is already computed in renderer

    // ----------------------------------------------
    // (2) Feature Prefiltering
    // ----------------------------------------------
    Flt_parameters p_pre = { .kc = 1., .kf = INFINITY, .tau = 0., .f=3, .r=5};
    buffer f_filtered, f_var_filtered;
    allocate_buffer(&f_filtered, img_width, img_height);
    allocate_buffer(&f_var_filtered, img_width, img_height);
    flt_buffer_basic(f_filtered, f, f, f_var, p_pre);
    flt_buffer_basic(f_var_filtered, f_var, f, f_var, p_pre);

    // ----------------------------------------------   
    // (3) Computation of Candidate Filters
    // ----------------------------------------------
    // (a) Candidate Filter: FIRST
    Flt_parameters p_r = { .kc = 0.45, .kf = 0.6, .tau = 0.001, .f = 1, .r=R};
    buffer r, d_r;
    allocate_buffer(&r, img_width, img_height);
    allocate_buffer(&d_r, img_width, img_height);
    flt(r, d_r, c, c, c_var, f_filtered, f_var_filtered, p_r);

    // (b) Candidate Filter: SECOND
    Flt_parameters p_g = { .kc = 0.45, .kf = 0.6, .tau = 0.001, .f = 3, .r=R};
    buffer g, d_g;
    allocate_buffer(&g, img_width, img_height);
    allocate_buffer(&d_g, img_width, img_height);
    flt(g, d_g, c, c, c_var, f_filtered, f_var_filtered, p_g);

    // (c) Candidate Filter: THIRD
    Flt_parameters p_b = { .kc = INFINITY, .kf = 0.6, .tau = 0.0001, .f = 1, .r=R};
    buffer b, d_b;
    allocate_buffer(&b, img_width, img_height);
    allocate_buffer(&d_b, img_width, img_height);
    flt(b, d_b, c, c, c_var, f_filtered, f_var_filtered, p_b);

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

    // (b) Filter error estimates
    Flt_parameters p_sure = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r=R};
    channel e_r, e_g, e_b;
    allocate_channel(&e_r, img_width, img_height);
    allocate_channel(&e_g, img_width, img_height);
    allocate_channel(&e_b, img_width, img_height);
    flt_channel_basic(e_r, sure_r, c, c_var, p_sure);
    flt_channel_basic(e_g, sure_g, c, c_var, p_sure);
    flt_channel_basic(e_b, sure_b, c, c_var, p_sure);

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
                sel_r[x][y] = e_r[x][y] < e_g[x][y] && e_r[x][y] < e_b[x][y] && d_r[0][x][y] < d_g[0][x][y];
                sel_g[x][y] = e_g[x][y] < e_r[x][y] && e_g[x][y] < e_b[x][y];
                sel_b[x][y] = e_b[x][y] < e_r[x][y] && e_g[x][y] < e_b[x][y];
        }
    }

    // ----------------------------------------------
    // (6) Filter Selection Maps
    // ----------------------------------------------
    Flt_parameters p_sel = { .kc = 1.0, .kf = INFINITY, .tau = 0.0001, .f = 1, .r=5};
    channel sel_r_filtered, sel_g_filtered,  sel_b_filtered;
    allocate_channel(&sel_r_filtered, img_width, img_height);
    allocate_channel(&sel_g_filtered, img_width, img_height);
    allocate_channel(&sel_b_filtered, img_width, img_height);
    flt_channel_basic(sel_r_filtered, sel_r, c, c_var, p_sel);
    flt_channel_basic(sel_g_filtered, sel_g, c, c_var, p_sel);
    flt_channel_basic(sel_b_filtered, sel_b, c, c_var, p_sel);

    // ----------------------------------------------
    // (7) Candidate Filter averaging
    // ----------------------------------------------
    buffer pass1;
    allocate_buffer(&pass1, img_width, img_height); 
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < img_width; x++){
            for (int y = 0; y < img_height; y++){
                pass1[i][x][y] = sel_r_filtered[x][y] * r[i][x][y] 
                                + sel_g_filtered[x][y] * g[i][x][y] 
                                + sel_b_filtered[x][y] * b[i][x][y];
            }
        }
    }


    // ----------------------------------------------
    // (8) Second Pass-Filtering
    // ----------------------------------------------
    // => not necessary since we use only one buffer at the moment
    // Therefore we can use pass1 as output
    out_img = pass1;

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




/* -------------------------------------------------------------------------
 * FUNCTION REGISTRATION
 * ------------------------------------------------------------------------- */ 
void register_functions()
{
    add_function(&basic_implementation, "Basic Implementation", 1);

}