#ifndef REGISTER_H
#define REGISTER_H

#include "../denoise.h"

void basic_implementation(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure2(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure_ILP_Test(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);

/* -------------------------------------------------------------------------
 * FUNCTION REGISTRATION
 * ------------------------------------------------------------------------- */ 
void register_functions()
{
    //add_function(&basic_implementation, "Basic Implementation", 1);
    //add_function(&basic_restructure, "Restructure (no precomputation)", 1);
    add_function(&basic_restructure2, "Restructure - With some precomputations (prefiltering, candidates)", 1);
    add_function(&basic_restructure_ILP_Test, "Restructure - With some precomputations (prefiltering, candidates)", 1);
}

#endif