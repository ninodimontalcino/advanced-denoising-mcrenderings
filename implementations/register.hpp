#ifndef REGISTER_H
#define REGISTER_H

#include "../denoise.h"

// 1D Arrays
void basic_implementation(scalar *out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H);
void basic_restructure1(scalar* out_img, scalar*  c, scalar*  c_var, scalar*  f, scalar*  f_var, int R, int W, int H);
void basic_restructure2(scalar* out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H);
void basic_restructure3(scalar* out_img, scalar* c, scalar* c_var, scalar* f, scalar* f_var, int R, int W, int H);

/* -------------------------------------------------------------------------
 * FUNCTION REGISTRATION
 * ------------------------------------------------------------------------- */ 
void register_functions()
{
    // add_function(&basic_implementation, "Basic Implementation", 1);
    add_function(&basic_restructure1, "Restructure - No Precomputations", 1);
    add_function(&basic_restructure2, "Restructure - With Precomputations (prefiltering, candidates)", 1);
    add_function(&basic_restructure3, "Restructure - With Precomputations (prefiltering, candidates) + ILP", 1);
}

#endif