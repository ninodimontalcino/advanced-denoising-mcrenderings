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

                    w = access_weight(weights, xp, yp, xq - xp + p.r, yq - yp + p.r, config);
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
                    wc = access_weight(weights, xp, yp, xq - xp + p.r, yq - yp + p.r, config);
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

void precompute_squared_difference(bufferweight output_numerator, bufferweight output_denominator, channel u, channel var_u, const int img_width, const int img_height, const int deltaMax)
{
    for (int xp = 0; xp < img_width; ++xp)
    {
        for (int yp = 0; yp < img_height; ++yp)
        {
            for (int deltaxq = -deltaMax; deltaxq <= deltaMax; ++deltaxq)
            {
                for (int deltayq = -deltaMax; deltayq <= deltaMax; ++deltayq)
                {
                    int xq = xp + deltaxq, yq = yp + deltayq;
                    if (xq < 0 || xq >= img_width || yq < 0 || yq >= img_height)
                        continue;
                    scalar sqdist = u[xp][yp] - u[xq][yq];
                    sqdist *= sqdist;
                    scalar var_cancel = var_u[xp][yp] + fmin(var_u[xp][yp], var_u[xq][yq]);
                    output_numerator[xp][yp][deltaxq + deltaMax][deltayq + deltaMax] = sqdist - var_cancel;
                    output_denominator[xp][yp][deltaxq + deltaMax][deltayq + deltaMax] = EPSILON + var_u[xp][yp] + var_u[xq][yq];
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
                                scalar current = (sq_dists[2 * i][xp + u][yp + v][deltaxq + deltaMax][deltayq + deltaMax] /
                                                  sq_dists[2 * i + 1][xp + u][yp + v][deltaxq + deltaMax][deltayq + deltaMax]);
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

void precompute_color_weights(bufferweightset allweights, scalar *allsums, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters *all_params, int n_params)
{
    for (int p = 0; p < n_params; ++p)
        allsums[p] = 0.f;

    bufferweightset sq_diffs;
    const int deltaMax = all_params[1].r + 7;
    allocate_buffer_weights(&sq_diffs, img_width, img_height, 6, deltaMax);
    for (int i = 0; i < 3; ++i)
        precompute_squared_difference(sq_diffs[2 * i], sq_diffs[2 * i + 1], u[i], var_u[i], img_width, img_height, deltaMax);

    bufferweightset diffs;
    const int r_max = all_params[0].r;
    allocate_buffer_weights(&diffs, img_width, img_height, 2, r_max);
    precompute_differences(diffs[0], diffs[1], sq_diffs, img_width, img_height, r_max, deltaMax);

    scalar wc;

    for (int xp = 2; xp < img_width - 2; ++xp)
    {
        for (int yp = 2; yp < img_height - 2; ++yp)
        {

            for (int xq = xp - r_max; xq <= xp + r_max; xq++)
            {
                for (int yq = yp - r_max; yq <= yp + r_max; yq++)
                {

                    for (int p = 0; p < n_params; ++p)
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
                            wc = exp(-fmax(0.f, diffs[0][xp][yp][xq - xp + r_max][yq - yp + r_max] / (27.f * all_params[p].kc * all_params[p].kc)));
                        }
                        else
                        {
                            wc = exp(-fmax(0.f, diffs[1][xp][yp][xq - xp + r_max][yq - yp + r_max] / (147.f * all_params[p].kc * all_params[p].kc)));
                        }
                        allweights[p][xp][yp][xq - xp + all_params[p].r][yq - yp + all_params[p].r] = wc;
                        allsums[p] += wc;
                    }
                }
            }
        }
    }

    free_buffer_weights(&sq_diffs, img_width, img_height, 6, deltaMax);
    free_buffer_weights(&diffs, img_width, img_height, 2, r_max);
}

void precompute_weights(bufferweightset allweights, scalar *allsums, buffer u, buffer var_u, buffer f, buffer var_f, int img_width, int img_height, Flt_parameters *all_params)
{
    // Computing gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for (int i = 0; i < NB_FEATURES; ++i)
        compute_gradient(gradients[i], f[i], 2, img_width, img_height); // 2 because we need almost the whole image for filter error

    precompute_color_weights(allweights, allsums, u, var_u, img_width, img_height, all_params, 5);

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

                    wc = allweights[0][xp][yp][xq - xp + p.r][yq - yp + p.r];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[0][xp][yp][xq - xp + p.r][yq - yp + p.r] = w;
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

                    wc = allweights[1][xp][yp][xq - xp + p.r][yq - yp + p.r];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[1][xp][yp][xq - xp + p.r][yq - yp + p.r] = w;
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

                    wc = allweights[2][xp][yp][xq - xp + p.r][yq - yp + p.r];
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    w = fmin(wc, wf);
                    allweights[2][xp][yp][xq - xp + p.r][yq - yp + p.r] = w;
                    allsums[2] += w;
                }
            }
        }
    }

    // Free memory
    free_buffer(&gradients, img_width);
}

scalar access_weight(bufferweightset weights, int xp, int yp, int xq, int yq, int config)
{
    return weights[config][xp][yp][xq][yq];
}