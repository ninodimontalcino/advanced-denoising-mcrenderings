#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fltopcount.hpp"
#include "flt.hpp"
#include "memory_mgmt.hpp"
#include <iostream>

void precompute_colors_pref(bufferweight weightpref, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters p)
{
    // compute colors weights for prefiltering in allweights[0] and allsums[0] for all_params[0]

    scalar wc;
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {
            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    weightpref[xp][yp][xq - xp + p.r][yq - yp + p.r] = wc; // translate in range (2r+1)(2r+1)
                }
            }
        }
    }
}

void flt_buffer_opcount(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height, bufferweight weights)
{
    scalar wc;

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < p.r + p.f; yp++)
            {
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
            }
        }
        for (int yp = p.r + p.f; yp < img_height - p.r + p.f; yp++)
        {
            for (int xp = 0; xp < p.r + p.f; xp++)
            {
                output[i][xp][yp] = input[i][xp][yp];
                output[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
            }
        }
    }

    scalar sum = 0;

    // General Pre-Filtering
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            // Init output to 0 => TODO: maybe we can do this with calloc
            for (int i = 0; i < 3; ++i)
                output[i][xp][yp] = 0.f;
            sum = 0.f;

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {
                    // Compute color Weight
                    wc = weights[xp][yp][xq - xp + p.r][yq - yp + p.r];
                    sum += wc;

                    // Add contribution term
                    for (int i = 0; i < 3; ++i)
                        output[i][xp][yp] += input[i][xq][yq] * wc;
                }
            }
            // Normalization step
            for (int i = 0; i < 3; ++i)
                output[i][xp][yp] /= (sum + EPSILON);
        }
    }
}

void flt_opcount(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights)
{
    scalar wc, wf, w;
    scalar sum_weights;

    // For edges, just copy in output the input
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < p.r + p.f; yp++)
            {
                out[i][xp][yp] = input[i][xp][yp];
                out[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][xp][img_height - yp - 1] = 0.f;
            }
        }
        for (int yp = p.r + p.f; yp < img_height - p.r + p.f; yp++)
        {
            for (int xp = 0; xp < p.r + p.f; xp++)
            {
                out[i][xp][yp] = input[i][xp][yp];
                out[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][img_width - xp - 1][yp] = 0.f;
            }
        }
    }

    // Real computation
    sum_weights = 0;
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            sum_weights = 0.f;

            for (int i = 0; i < 3; ++i)
                out[i][xp][yp] = 0;

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {

                    w = weights[xp][yp][xq - xp + p.r][yq - yp + p.r][config];
                    sum_weights += w;

                    for (int i = 0; i < 3; ++i)
                    {
                        out[i][xp][yp] += input[i][xq][yq] * w;
                    }
                }
            }

            for (int i = 0; i < 3; ++i)
            {
                out[i][xp][yp] /= (sum_weights + EPSILON);

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i][xp][yp] = 0.f;
            }
        }
    }
}

void flt_channel_opcount(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights)
{

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++)
    {
        for (int yp = 0; yp < p.r + p.f; yp++)
        {
            output[xp][yp] = input[xp][yp];
            output[xp][img_height - yp - 1] = input[xp][img_height - yp - 1];
        }
    }
    for (int yp = p.r + p.f; yp < img_height - p.r + p.f; yp++)
    {
        for (int xp = 0; xp < p.r + p.f; xp++)
        {
            output[xp][yp] = input[xp][yp];
            output[img_width - xp - 1][yp] = input[img_width - xp - 1][yp];
        }
    }

    // General Pre-Filtering
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            output[xp][yp] = 0.f;

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {

                    // Compute color Weight
                    wc = weights[xp][yp][xq - xp + p.r][yq - yp + p.r][config];
                    sum_weights += wc;

                    // Add contribution term
                    output[xp][yp] += input[xq][yq] * wc;
                }
            }

            // Normalization step
            output[xp][yp] /= (sum_weights + EPSILON);
        }
    }
}

void precompute_squared_difference(bufferweightset sq_dists, buffer u, buffer var_u, const int img_width, const int img_height, const int deltaMax)
{
    for(int i=0;i<3;++i) {
        for (int xp = 0; xp < img_width; ++xp)
        {
            for (int yp = 0; yp < img_height; ++yp)
            {
                for (int deltaxq = -deltaMax; deltaxq <= deltaMax; ++deltaxq)
                {
                    for (int deltayq = 0; deltayq <= deltaMax; ++deltayq)
                    {
                        int xq = xp + deltaxq, yq = yp + deltayq;
                        if (xq < 0 || xq >= img_width || yq < 0 || yq >= img_height) // deal with border
                            continue;
                        
                        scalar sqdist = u[i][xp][yp] - u[i][xq][yq];
                        sqdist *= sqdist;
                        scalar varqp = fmin(var_u[i][xp][yp], var_u[i][xq][yq]);
                        scalar var_cancel_p = var_u[i][xp][yp] + varqp;
                        scalar var_cancel_q = var_u[i][xq][yq] + varqp;

                        sq_dists[xp][yp][deltaxq + deltaMax][deltayq + deltaMax][2*i] = sqdist - var_cancel_p;
                        sq_dists[xq][yq][-deltaxq + deltaMax][-deltayq + deltaMax][2*i] = sqdist - var_cancel_q;

                        sq_dists[xp][yp][deltaxq + deltaMax][deltayq + deltaMax][2*i+1] = EPSILON + var_u[i][xp][yp] + var_u[i][xq][yq];
                        sq_dists[xq][yq][-deltaxq + deltaMax][-deltayq + deltaMax][2*i+1] = EPSILON + var_u[i][xp][yp] + var_u[i][xq][yq];

                    }
                }
            }
        }
    }
}

void precompute_differences(bufferweight diff1, bufferweight diff2, bufferweightset sq_dists, const int img_width, const int img_height, const int maxR, const int deltaMax)
{
    for (int xp = 0; xp < img_width; ++xp)
    {
        for (int yp = 0; yp < img_height; ++yp)
        {
            for (int deltaxq = -maxR; deltaxq <= maxR; deltaxq++)
            {
                for (int deltayq = -maxR; deltayq <= maxR; deltayq++)
                {
                    int xq = xp + deltaxq, yq = yp + deltayq;
                    if (xq < 0 || xq >= img_width || yq < 0 || yq >= img_height)
                        continue;
                    // Diff1 is the sum for f=1, diff2 for f=3
                    diff1[xp][yp][deltaxq + maxR][deltayq + maxR] = 0;
                    diff2[xp][yp][deltaxq + maxR][deltayq + maxR] = 0;
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int u = -3; u <= 3; ++u)
                        {
                            for (int v = -3; v <= 3; ++v)
                            {
                                if (xp + u < 0 || xp + u >= img_height || yp + v < 0 || yp + v >= img_height ||
                                    xq + u < 0 || xq + u >= img_height || yq + v < 0 || yq + v >= img_height)
                                    continue;
                                scalar current = (sq_dists[xp + u][yp + v][deltaxq + deltaMax][deltayq + deltaMax][2 * i] /
                                                  sq_dists[xp + u][yp + v][deltaxq + deltaMax][deltayq + deltaMax][2 * i + 1]);
                                diff2[xp][yp][deltaxq + maxR][deltayq + maxR] += current;
                                if (u >= -1 && u <= 1 && v >= -1 && v <= 1)
                                {
                                    diff1[xp][yp][deltaxq + maxR][deltayq + maxR] += current;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void precompute_color_weights(bufferweightset allweights, scalar *allsums, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters *all_params, const int offset)
{
    for (int p = 0; p < 5; ++p)
        allsums[p] = 0.f;
    
    precompute_squared_difference(allweights, u, var_u, img_width, img_height, offset);

    bufferweight diffs1, diffs2;
    const int r_max = all_params[0].r;
    allocate_buffer_weights(&diffs1, img_width, img_height, r_max);
    allocate_buffer_weights(&diffs2, img_width, img_height, r_max);
    precompute_differences(diffs1, diffs2, allweights, img_width, img_height, r_max, offset);

    scalar wc;

    // precompute division
    scalar f1 = 1.f / 108.f; // 108 = 3(2*1+1)^2 * 2^2
    scalar f3 = 1.f / 147.f; // 147 = 3(2^3+1)^2 * 1^2

    for (int xp = 2; xp < img_width - 2; ++xp)
    {
        for (int yp = 2; yp < img_height - 2; ++yp)
        {

            for (int xq = xp - r_max; xq <= xp + r_max; xq++)
            {
                for (int yq = yp - r_max; yq <= yp + r_max; yq++)
                {

                    for (int p = 0; p < 5; ++p)
                    {
                        if ((xp < all_params[p].r + all_params[p].f || xp >= img_width - all_params[p].r - all_params[p].f) ||
                            (yp < all_params[p].r + all_params[p].f || yp >= img_height - all_params[p].r - all_params[p].f) ||
                            (xq < xp - all_params[p].r || xq > xp + all_params[p].r) ||
                            (yq < yp - all_params[p].r || yq > yp + all_params[p].r))
                            continue;
                        if (all_params[p].kc == INFINITY)
                        {
                            wc = 1;
                        }
                        else if (all_params[p].f == 1)
                        {
                            wc = exp(-fmax(0.f, diffs1[xp][yp][xq - xp + r_max][yq - yp + r_max] / (27.f * all_params[p].kc * all_params[p].kc)));
                        }
                        else
                        {
                            wc = exp(-fmax(0.f, diffs2[xp][yp][xq - xp + r_max][yq - yp + r_max] / (147.f * all_params[p].kc * all_params[p].kc)));
                        }
                        allweights[xp][yp][xq - xp + all_params[p].r][yq - yp + all_params[p].r][p] = wc;
                        allsums[p] += wc;
                    }
                }
            }
        }
    }

    free_buffer_weights(&diffs1, img_width, img_height, r_max);
    free_buffer_weights(&diffs2, img_width, img_height, r_max);
}

void precompute_weights(bufferweightset allweights, scalar *allsums, buffer u, buffer var_u, buffer f, buffer var_f, int img_width, int img_height, Flt_parameters *all_params, const int offset)
{
    // Computing gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for (int i = 0; i < NB_FEATURES; ++i)
        compute_gradient(gradients[i], f[i], 2, img_width, img_height); // 2 because we need almost the whole image for filter error

    precompute_color_weights(allweights, allsums, u, var_u, img_width, img_height, all_params, offset);

    scalar wc, wf, w;
    Flt_parameters p;

    // FIRST candidate
    p = all_params[0];
    allsums[0] = 0.f;
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {

                    wc = allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][0];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][0] = w;
                    allsums[0] += w;
                }
            }
        }
    }

    // SECOND candidate
    p = all_params[1];
    allsums[1] = 0.f;
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {

                    wc = allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][1];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][1] = w;
                    allsums[1] += w;
                }
            }
        }
    }

    // THIRD candidate
    p = all_params[2];
    allsums[2] = 0.f;
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            for (int xq = xp - p.r; xq <= xp + p.r; xq++)
            {
                for (int yq = yp - p.r; yq <= yp + p.r; yq++)
                {

                    wc = allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][2];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[xp][yp][xq - xp + p.r][yq - yp + p.r][2] = w;
                    allsums[2] += w;
                }
            }
        }
    }

    // Free memory
    free_buffer(&gradients, img_width);
}