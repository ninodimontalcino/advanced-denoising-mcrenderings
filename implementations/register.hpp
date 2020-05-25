#ifndef REGISTER_H
#define REGISTER_H

#include "../denoise.h"

void basic_implementation(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure1(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure2(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure3(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure4(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure6(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure7(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void basic_restructure8(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);

/* -------------------------------------------------------------------------
 * FUNCTION REGISTRATION
 * ------------------------------------------------------------------------- */ 
void register_functions()
{
    //add_function(&basic_implementation, "Basic Implementation", 1);
    //add_function(&basic_restructure, "Restructure", 1);
    //add_function(&basic_restructure1, "Restructure + ILP/SSA", 1);
    //add_function(&basic_restructure2, "Restructure + Precomputations", 1);
    add_function(&basic_restructure3, "Restructure + Precomputations + ILP/SSA", 1);
    add_function(&basic_restructure4, "Restructure + Precomputations + ILP/SSA + Vectorization", 1);
    add_function(&basic_restructure6, "Restructure + Precomputations + ILP/SSA + Vectorization + Blocking Preparation", 1);
    add_function(&basic_restructure7, "Restructure + Precomputations + ILP/SSA + Vectorization + Blocking Clean", 1);
    add_function(&basic_restructure8, "Restructure + Precomputations + ILP/SSA + Vectorization + Blocking Clean + removed precompute", 1);
}

#endif