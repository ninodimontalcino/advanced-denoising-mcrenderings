#ifndef REGISTER_H
#define REGISTER_H

#include "../denoise.h"

void basic_implementation(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);
void optopcount(buffer out_img, buffer c, buffer c_var, buffer f, buffer f_var, int R, int img_width, int img_height);

/* -------------------------------------------------------------------------
 * FUNCTION REGISTRATION
 * ------------------------------------------------------------------------- */ 
void register_functions()
{
    //add_function(&basic_implementation, "Basic Implementation", 1);
    add_function(&optopcount, "Optimization of ops count", 1);
}

#endif