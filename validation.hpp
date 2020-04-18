#ifndef VALIDATION_H
#define VALIDATION_H

#include "flt.hpp"


// Right now this is a random value, used to compare if 2 floats are equal or not
#define FLOAT_TOLERANCE 0.00001

/* -------------------------------------------------------
    Root Mean Squared Error (RMSE) Computation

    Parameters:
        - denoised (buffer):    buffer containing denoised image
        - gt (buffer):          buffer containing ground truth image
        - img_width (int:       image width
        - img_height (int):     image height

    Returns:
        - rmse (scalar):         RMSE between denoised image and ground truth image

*/
scalar rmse(buffer denoised, buffer gt, int img_width, int img_height);


bool compare_scalar(scalar x, scalar y);
bool compare_buffers(buffer buf1, buffer buf2, int img_width, int img_height);


#endif //VALIDATION_H