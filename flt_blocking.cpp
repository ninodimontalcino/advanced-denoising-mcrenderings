#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "flt_blocking.hpp"
#include "memory_mgmt.hpp"

void sure_all_blocking(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int img_width, int img_height)
{

    scalar d_r, d_g, d_b, v;

    // Sum over color channels
    for (int i = 0; i < 3; i++)
    {
        for (int x = 0; x < img_width; x++)
        {
            for (int y = 0; y < img_height; y++)
            {

                // Calculate terms
                d_r = cand_r[i][x][y] - c[i][x][y];
                d_r *= d_r;
                d_g = cand_g[i][x][y] - c[i][x][y];
                d_g *= d_g;
                d_b = cand_b[i][x][y] - c[i][x][y];
                d_b *= d_b;
                v = c_var[i][x][y];
                v *= v;

                // Store sure error estimate
                sure[0][x][y] += d_r - v;
                sure[1][x][y] += d_g - v;
                sure[2][x][y] += d_b - v;
            }
        }
    }
}

void filtering_basic_blocking(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int img_width, int img_height, int blocks_width_size, int blocks_height_size)
{

    // Handling Inner Part
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height);
    allocate_channel(&temp2, img_width, img_height);

    // Precompute size of neighbourhood
    scalar neigh = 3 * (2 * p.f + 1) * (2 * p.f + 1);

    const int nb_Blocks_width = (img_width - 2 * (p.r + p.f)) / (blocks_width_size - 2 * (p.r + p.f));
    const int nb_Blocks_height = (img_height - 2 * (p.r + p.f)) / (blocks_height_size - 2 * (p.r + p.f));

    int begin_x = 0;
    int end_x = img_width;
    int begin_y = 0;
    int end_y = img_height;

    for (int b_w = 0; b_w < nb_Blocks_width; ++b_w)
    {
        begin_x = b_w * (blocks_width_size - 2 * (p.r + p.f));
        end_x = begin_x + blocks_width_size;
        for (int b_h = 0; b_h < nb_Blocks_height; ++b_h)
        {
            begin_y = b_h * (blocks_height_size - 2 * (p.r + p.f));
            end_y = begin_y + blocks_height_size;
            // Covering the neighbourhood
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {
                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r; yp < end_y - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (c_var[i][xp][yp] + c_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * input[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                {
                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                    }
                }
            }
        }

        if (end_y < img_height)
        {
            begin_y = nb_Blocks_height * (blocks_height_size - 2 * (p.r + p.f));
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {
                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r; yp < img_height - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (c_var[i][xp][yp] + c_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * input[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {
                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                    }
                }
            }
        }
    }

    if (end_x < img_width)
    {
        begin_x = nb_Blocks_width * (blocks_width_size - 2 * (p.r + p.f));
        for (int b_h = 0; b_h < nb_Blocks_height; ++b_h)
        {
            begin_y = b_h * (blocks_height_size - 2 * (p.r + p.f));
            end_y = begin_y + blocks_height_size;
            // Covering the neighbourhood
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {
                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r; yp < end_y - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (c_var[i][xp][yp] + c_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * input[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                {
                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                    }
                }
            }
        }

        if (end_y < img_height)
        {
            begin_y = nb_Blocks_height * (blocks_height_size - 2 * (p.r + p.f));
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {
                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r; yp < img_height - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (c_var[i][xp][yp] + c_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * input[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {
                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                    }
                }
            }
        }
    
    }

    // Handline Border Cases
    // ---------------------
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
        for (int xp = 0; xp < p.r + p.f; xp++)
        {
            for (int yp = p.r + p.f; yp < img_height - p.r - p.f; yp++)
            {
                output[i][xp][yp] = input[i][xp][yp];
                output[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
            }
        }
    }

    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width);
}

void feature_prefiltering_blocking(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height, int blocks_width_size, int blocks_height_size)
{

    // Handling Inner Part
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height);
    allocate_channel(&temp2, img_width, img_height);

    // Precompute size of neighbourhood
    scalar neigh = 3 * (2 * p.f + 1) * (2 * p.f + 1);

    const int nb_Blocks_width = (img_width - 2 * (p.r + p.f)) / (blocks_width_size - 2 * (p.r + p.f));
    const int nb_Blocks_height = (img_height - 2 * (p.r + p.f)) / (blocks_height_size - 2 * (p.r + p.f));

    int begin_x = 0;
    int end_x = 0;
    int begin_y = 0;
    int end_y = 0;

    // Covering the neighbourhood
    for (int b_w = 0; b_w < nb_Blocks_width; ++b_w)
    {
        begin_x = b_w * (blocks_width_size - 2 * (p.r + p.f));
        end_x = begin_x + blocks_width_size;
        for (int b_h = 0; b_h < nb_Blocks_height; ++b_h)
        {
            begin_y = b_h * (blocks_height_size - 2 * (p.r + p.f));
            end_y = begin_y + blocks_height_size;
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {

                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        if (r_x == 0 && r_y == 0 && b_h == 0)
                        {
                            // printf("%d\n", xp);
                        }
                        for (int yp = begin_y + p.r; yp < end_y - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (features_var[i][xp][yp] + features_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            // if (r_x==0 && r_y==0 && b_w==0 && xp == begin_x + p.r+p.f){
                            //     printf("%d\n", yp);
                            // }

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * features[i][xq][yq];
                                output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                {

                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                        output_var[i][xp][yp] /= w;
                    }
                }
            }
        }
        if (end_y != img_height)
        {
            begin_y = nb_Blocks_height * (blocks_height_size - 2 * (p.r + p.f));
            // printf("coucou %d\n", end_y);
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {

                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        if (r_x == 0 && r_y == 0)
                        {
                            // printf("%d\n", xp);
                        }
                        for (int yp = begin_y + p.r; yp < img_height - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (features_var[i][xp][yp] + features_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < end_x - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {
                            // if (r_x==0 && r_y==0 && b_w==0 && xp == begin_x + p.r+p.f){
                            //     printf("%d\n", yp);
                            // }

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * features[i][xq][yq];
                                output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < end_x - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {

                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                        output_var[i][xp][yp] /= w;
                    }
                }
            }
        }
    }

    if (end_x != img_width)
    {
        begin_x = nb_Blocks_width * (blocks_width_size - 2 * (p.r + p.f));
        for (int b_h = 0; b_h < nb_Blocks_height; ++b_h)
        {
            begin_y = b_h * (blocks_height_size - 2 * (p.r + p.f));
            end_y = begin_y + blocks_height_size;
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {

                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        if (r_x == 0 && r_y == 0 && b_h == 0)
                        {
                            // printf("%d\n", xp);
                        }
                        for (int yp = begin_y + p.r; yp < end_y - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (features_var[i][xp][yp] + features_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                        {

                            // if (r_x==0 && r_y==0 && b_w==0 && xp == begin_x + p.r+p.f){
                            //     printf("%d\n", yp);
                            // }

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * features[i][xq][yq];
                                output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < end_y - p.r - p.f; ++yp)
                {

                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                        output_var[i][xp][yp] /= w;
                    }
                }
            }
        }
        if (end_y != img_height)
        {
            begin_y = nb_Blocks_height * (blocks_height_size - 2 * (p.r + p.f));
            // printf("coucou %d\n", end_y);
            for (int r_x = -p.r; r_x <= p.r; r_x++)
            {
                for (int r_y = -p.r; r_y <= p.r; r_y++)
                {

                    // Compute Color Weight for all pixels with fixed r
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        if (r_x == 0 && r_y == 0)
                        {
                            // printf("%d\n", xp);
                        }
                        for (int yp = begin_y + p.r; yp < img_height - p.r; ++yp)
                        {

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar distance = 0;
                            for (int i = 0; i < 3; i++)
                            {
                                scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                                sqdist *= sqdist;
                                scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                                scalar normalization = EPSILON + k_c_squared * (features_var[i][xp][yp] + features_var[i][xq][yq]);
                                distance += (sqdist - var_cancel) / normalization;
                            }

                            temp[xp][yp] = distance;
                        }
                    }

                    // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
                    // (1) Convolve along height
                    for (int xp = begin_x + p.r; xp < img_width - p.r; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp[xp][yp + k];
                            }
                            temp2[xp][yp] = sum;
                        }
                    }

                    // (2) Convolve along width including weighted contribution
                    for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
                    {
                        for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                        {
                            // if (r_x==0 && r_y==0 && b_w==0 && xp == begin_x + p.r+p.f){
                            //     printf("%d\n", yp);
                            // }

                            int xq = xp + r_x;
                            int yq = yp + r_y;

                            scalar sum = 0.f;
                            for (int k = -p.f; k <= p.f; k++)
                            {
                                sum += temp2[xp + k][yp];
                            }
                            scalar weight = exp(-fmax(0.f, (sum / neigh)));
                            weight_sum[xp][yp] += weight;

                            for (int i = 0; i < 3; i++)
                            {
                                output[i][xp][yp] += weight * features[i][xq][yq];
                                output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                            }
                        }
                    }
                }
            }

            // Final Weight Normalization
            for (int xp = begin_x + p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = begin_y + p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {

                    scalar w = weight_sum[xp][yp];
                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] /= w;
                        output_var[i][xp][yp] /= w;
                    }
                }
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < p.r + p.f; yp++)
            {
                output[i][xp][yp] = features[i][xp][yp];
                output[i][xp][img_height - yp - 1] = features[i][xp][img_height - yp - 1];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][img_height - yp - 1] = features_var[i][xp][img_height - yp - 1];
            }
        }
        for (int xp = 0; xp < p.r + p.f; xp++)
        {
            for (int yp = p.r + p.f; yp < img_height - p.r - p.f; yp++)
            {
                output[i][xp][yp] = features[i][xp][yp];
                output[i][img_width - xp - 1][yp] = features[i][img_width - xp - 1][yp];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][img_height - yp - 1] = features_var[i][xp][img_height - yp - 1];
            }
        }
    }

    // Free memory
    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width);
}

void candidate_filtering_blocking(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height)
{

    // Handling Inner Part
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    scalar k_f_squared = p.kf * p.kf;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height);
    allocate_channel(&temp2, img_width, img_height);

    // Init feature weights channel
    channel feature_weights;
    allocate_channel(&feature_weights, img_width, img_height);

    // Compute gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for (int i = 0; i < NB_FEATURES; ++i)
    {
        compute_gradient(gradients[i], features[i], p.r + p.f, img_width, img_height);
    }

    // Precompute size of neighbourhood
    scalar neigh = 3 * (2 * p.f + 1) * (2 * p.f + 1);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++)
    {
        for (int r_y = -p.r; r_y <= p.r; r_y++)
        {

            // Compute Color Weight for all pixels with fixed r
            for (int xp = p.r; xp < img_width - p.r; ++xp)
            {
                for (int yp = p.r; yp < img_height - p.r; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared * (color_var[i][xp][yp] + color_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp][yp] = distance;
                }
            }

            // Compute features
            for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute feature weight
                    scalar df = 0.f;
                    for (int j = 0; j < NB_FEATURES; ++j)
                    {
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar normalization = k_f_squared * fmax(p.tau, fmax(features_var[j][xp][yp], gradients[j][xp][yp]));
                        df = fmax(df, (sqdist - var_cancel) / normalization);
                    }
                    feature_weights[xp][yp] = exp(-df);
                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for (int xp = p.r; xp < img_width - p.r; ++xp)
            {
                for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {
                    scalar sum = 0.f;
                    for (int k = -p.f; k <= p.f; k++)
                    {
                        sum += temp[xp][yp + k];
                    }
                    temp2[xp][yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
            {
                for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k = -p.f; k <= p.f; k++)
                    {
                        sum += temp2[xp + k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh)));

                    scalar weight = fmin(color_weight, feature_weights[xp][yp]);
                    weight_sum[xp][yp] += weight;

                    for (int i = 0; i < 3; i++)
                    {
                        output[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization
    for (int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp)
    {
        for (int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp)
        {

            scalar w = weight_sum[xp][yp];
            for (int i = 0; i < 3; i++)
            {
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < p.r + p.f; yp++)
            {
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for (int xp = 0; xp < p.r + p.f; xp++)
        {
            for (int yp = p.r + p.f; yp < img_height - p.r - p.f; yp++)
            {
                output[i][xp][yp] = color[i][xp][yp];
                output[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Free memory
    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width);
    free_channel(&feature_weights, img_width);
    free_buffer(&gradients, img_width);
}

// ====================================================================================================================================================================================================================================
// !!! TO BE IMPROVED => still more precomputations possible
// ====================================================================================================================================================================================================================================

void candidate_filtering_all_blocking(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters *p, int img_width, int img_height)
{

    // Get parameters
    int f_r = p[0].f;
    int f_g = p[1].f;
    int f_b = p[2].f;
    scalar tau_r = p[0].tau;
    scalar tau_g = p[1].tau;
    scalar tau_b = p[2].tau;
    scalar k_c_squared_r = p[0].kc * p[0].kc;
    scalar k_f_squared_r = p[0].kf * p[0].kf;
    scalar k_c_squared_g = p[1].kc * p[1].kc;
    scalar k_f_squared_g = p[1].kf * p[1].kf;
    scalar k_f_squared_b = p[2].kf * p[2].kf;

    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;

    // Handling Inner Part
    // -------------------

    // Allocate buffer weights_sum for normalizing
    buffer weight_sum;
    allocate_buffer_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp;
    channel temp2_r;
    channel temp2_g;
    allocate_channel(&temp, img_width, img_height);
    allocate_channel(&temp2_r, img_width, img_height);
    allocate_channel(&temp2_g, img_width, img_height);

    // Allocate feature weights buffer
    channel features_weights_r;
    channel features_weights_b;
    allocate_channel(&features_weights_r, img_width, img_height);
    allocate_channel(&features_weights_b, img_width, img_height);

    // Compute gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for (int i = 0; i < NB_FEATURES; ++i)
    {
        compute_gradient(gradients[i], features[i], R + f_min, img_width, img_height);
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3 * (2 * f_r + 1) * (2 * f_r + 1);
    scalar neigh_g = 3 * (2 * f_g + 1) * (2 * f_g + 1);
    scalar neigh_b = 3 * (2 * f_b + 1) * (2 * f_b + 1);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++)
    {
        for (int r_y = -R; r_y <= R; r_y++)
        {

            // Compute Color Weight for all pixels with fixed r
            for (int xp = R; xp < img_width - R; ++xp)
            {
                for (int yp = R; yp < img_height - R; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance_r = 0.f;

                    for (int i = 0; i < 3; i++)
                    {
                        scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                        scalar normalization_r = EPSILON + k_c_squared_r * (var_term);
                        scalar dist_var = sqdist - var_cancel;
                        distance_r += (dist_var) / normalization_r;
                    }

                    temp[xp][yp] = distance_r;
                }
            }

            // Precompute feature weights
            for (int xp = R + f_min; xp < img_width - R - f_min; ++xp)
            {
                for (int yp = R + f_min; yp < img_height - R - f_min; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r = 0.f;
                    scalar df_b = 0.f;

                    for (int j = 0; j < NB_FEATURES; ++j)
                    {
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar var_max = fmax(features_var[j][xp][yp], gradients[j][xp][yp]);
                        scalar normalization_r = k_f_squared_r * fmax(tau_r, var_max);
                        scalar normalization_b = k_f_squared_b * fmax(tau_b, var_max);
                        scalar dist_var = sqdist - var_cancel;
                        df_r = fmax(df_r, (dist_var) / normalization_r);
                        df_b = fmax(df_b, (dist_var) / normalization_b);
                    }

                    features_weights_r[xp][yp] = exp(-df_r);
                    features_weights_b[xp][yp] = exp(-df_b);
                }
            }

            // Next Steps: Box-Filtering for Patch Contribution
            // => Use Box-Filter Seperability => linear scans of data

            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for (int xp = R; xp < img_width - R; ++xp)
            {
                for (int yp = R + f_r; yp < img_height - R - f_r; ++yp)
                {
                    scalar sum_r = 0.f;
                    for (int k = -f_r; k <= f_r; k++)
                    {
                        sum_r += temp[xp][yp + k];
                    }
                    temp2_r[xp][yp] = sum_r;
                }
            }

            // (2) Convolve along width including weighted contribution
            for (int xp = R + f_r; xp < img_width - R - f_r; ++xp)
            {
                for (int yp = R + f_r; yp < img_height - R - f_r; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k = -f_r; k <= f_r; k++)
                    {
                        sum += temp2_r[xp + k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_r)));

                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp][yp]);
                    weight_sum[0][xp][yp] += weight;

                    for (int i = 0; i < 3; i++)
                    {
                        output_r[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for (int xp = R; xp < img_width - R; ++xp)
            {
                for (int yp = R + f_g; yp < img_height - R - f_g; ++yp)
                {
                    scalar sum_g = 0.f;
                    for (int k = -f_g; k <= f_g; k++)
                    {
                        sum_g += temp[xp][yp + k];
                    }
                    temp2_g[xp][yp] = sum_g;
                }
            }

            // (2) Convolve along width including weighted contribution
            for (int xp = R + f_g; xp < img_width - R - f_g; ++xp)
            {
                for (int yp = R + f_g; yp < img_height - R - f_g; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k = -f_g; k <= f_g; k++)
                    {
                        sum += temp2_g[xp + k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_g)));

                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp][yp]);
                    weight_sum[1][xp][yp] += weight;

                    for (int i = 0; i < 3; i++)
                    {
                        output_g[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for (int xp = R + f_b; xp < img_width - R - f_b; ++xp)
            {
                for (int yp = R + f_b; yp < img_height - R - f_b; ++yp)
                {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight = features_weights_b[xp][yp];
                    weight_sum[2][xp][yp] += weight;

                    for (int i = 0; i < 3; i++)
                    {
                        output_b[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization R
    for (int xp = R + f_r; xp < img_width - R - f_r; ++xp)
    {
        for (int yp = R + f_r; yp < img_height - R - f_r; ++yp)
        {

            scalar w = weight_sum[0][xp][yp];
            for (int i = 0; i < 3; i++)
            {
                output_r[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization G
    for (int xp = R + f_g; xp < img_width - R - f_g; ++xp)
    {
        for (int yp = R + f_g; yp < img_height - R - f_g; ++yp)
        {

            scalar w = weight_sum[1][xp][yp];
            for (int i = 0; i < 3; i++)
            {
                output_g[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization B
    for (int xp = R + f_b; xp < img_width - R - f_b; ++xp)
    {
        for (int yp = R + f_b; yp < img_height - R - f_b; ++yp)
        {

            scalar w = weight_sum[2][xp][yp];
            for (int i = 0; i < 3; i++)
            {
                output_b[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < R + f_r; yp++)
            {
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for (int xp = 0; xp < R + f_r; xp++)
        {
            for (int yp = R + f_r; yp < img_height - R - f_r; yp++)
            {

                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++)
    {
        for (int xp = 0; xp < img_width; xp++)
        {
            for (int yp = 0; yp < R + f_g; yp++)
            {
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for (int xp = 0; xp < R + f_g; xp++)
        {
            for (int yp = R + f_g; yp < img_height - R - f_g; yp++)
            {
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Free memory
    free_buffer(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2_r, img_width);
    free_channel(&temp2_g, img_width);
    free_channel(&features_weights_r, img_width);
    free_channel(&features_weights_b, img_width);
    free_buffer(&gradients, img_width);
}
