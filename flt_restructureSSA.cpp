#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"

// Get parameters
#define F_R 1
#define F_G 3
#define F_B 1
#define TAU_R 0.001
#define TAU_G 0.001
#define TAU_B 0.0001
#define KC_SQUARED_R 4.0
#define KF_SQUARED_R 0.36
#define KC_SQUARED_G 4.0
#define KF_SQUARED_G 0.36
#define KF_SQUARED_B 0.36

void compute_gradients_SSA(scalar *gradients, scalar *features, const int W, const int H, const int R) {
    scalar diffL_A0, diffR_A0, diffU_A0, diffD_A0;
    scalar diffL_A1, diffR_A1, diffU_A1, diffD_A1;
    scalar diffL_A2, diffR_A2, diffU_A2, diffD_A2;
    scalar diffsqL_A0, diffsqR_A0, diffsqU_A0, diffsqD_A0;
    scalar diffsqL_A1, diffsqR_A1, diffsqU_A1, diffsqD_A1;
    scalar diffsqL_A2, diffsqR_A2, diffsqU_A2, diffsqD_A2;
    scalar gradH_A0, gradV_A0;
    scalar gradH_A1, gradV_A1;
    scalar gradH_A2, gradV_A2;
    scalar diffL_B0, diffR_B0, diffU_B0, diffD_B0;
    scalar diffL_B1, diffR_B1, diffU_B1, diffD_B1;
    scalar diffL_B2, diffR_B2, diffU_B2, diffD_B2;
    scalar diffsqL_B0, diffsqR_B0, diffsqU_B0, diffsqD_B0;
    scalar diffsqL_B1, diffsqR_B1, diffsqU_B1, diffsqD_B1;
    scalar diffsqL_B2, diffsqR_B2, diffsqU_B2, diffsqD_B2;
    scalar gradH_B0, gradV_B0;
    scalar gradH_B1, gradV_B1;
    scalar gradH_B2, gradV_B2;
    scalar diffL_C0, diffR_C0, diffU_C0, diffD_C0;
    scalar diffL_C1, diffR_C1, diffU_C1, diffD_C1;
    scalar diffL_C2, diffR_C2, diffU_C2, diffD_C2;
    scalar diffsqL_C0, diffsqR_C0, diffsqU_C0, diffsqD_C0;
    scalar diffsqL_C1, diffsqR_C1, diffsqU_C1, diffsqD_C1;
    scalar diffsqL_C2, diffsqR_C2, diffsqU_C2, diffsqD_C2;
    scalar gradH_C0, gradV_C0;
    scalar gradH_C1, gradV_C1;
    scalar gradH_C2, gradV_C2;
    scalar diffL_D0, diffR_D0, diffU_D0, diffD_D0;
    scalar diffL_D1, diffR_D1, diffU_D1, diffD_D1;
    scalar diffL_D2, diffR_D2, diffU_D2, diffD_D2;
    scalar diffsqL_D0, diffsqR_D0, diffsqU_D0, diffsqD_D0;
    scalar diffsqL_D1, diffsqR_D1, diffsqU_D1, diffsqD_D1;
    scalar diffsqL_D2, diffsqR_D2, diffsqU_D2, diffsqD_D2;
    scalar gradH_D0, gradV_D0;
    scalar gradH_D1, gradV_D1;
    scalar gradH_D2, gradV_D2;
    scalar diffL_E0, diffR_E0, diffU_E0, diffD_E0;
    scalar diffL_E1, diffR_E1, diffU_E1, diffD_E1;
    scalar diffL_E2, diffR_E2, diffU_E2, diffD_E2;
    scalar diffsqL_E0, diffsqR_E0, diffsqU_E0, diffsqD_E0;
    scalar diffsqL_E1, diffsqR_E1, diffsqU_E1, diffsqD_E1;
    scalar diffsqL_E2, diffsqR_E2, diffsqU_E2, diffsqD_E2;
    scalar gradH_E0, gradV_E0;
    scalar gradH_E1, gradV_E1;
    scalar gradH_E2, gradV_E2;
    scalar diffL_F0, diffR_F0, diffU_F0, diffD_F0;
    scalar diffL_F1, diffR_F1, diffU_F1, diffD_F1;
    scalar diffL_F2, diffR_F2, diffU_F2, diffD_F2;
    scalar diffsqL_F0, diffsqR_F0, diffsqU_F0, diffsqD_F0;
    scalar diffsqL_F1, diffsqR_F1, diffsqU_F1, diffsqD_F1;
    scalar diffsqL_F2, diffsqR_F2, diffsqU_F2, diffsqD_F2;
    scalar gradH_F0, gradV_F0;
    scalar gradH_F1, gradV_F1;
    scalar gradH_F2, gradV_F2;
    scalar diffL_G0, diffR_G0, diffU_G0, diffD_G0;
    scalar diffL_G1, diffR_G1, diffU_G1, diffD_G1;
    scalar diffL_G2, diffR_G2, diffU_G2, diffD_G2;
    scalar diffsqL_G0, diffsqR_G0, diffsqU_G0, diffsqD_G0;
    scalar diffsqL_G1, diffsqR_G1, diffsqU_G1, diffsqD_G1;
    scalar diffsqL_G2, diffsqR_G2, diffsqU_G2, diffsqD_G2;
    scalar gradH_G0, gradV_G0;
    scalar gradH_G1, gradV_G1;
    scalar gradH_G2, gradV_G2;
    scalar diffL_H0, diffR_H0, diffU_H0, diffD_H0;
    scalar diffL_H1, diffR_H1, diffU_H1, diffD_H1;
    scalar diffL_H2, diffR_H2, diffU_H2, diffD_H2;
    scalar diffsqL_H0, diffsqR_H0, diffsqU_H0, diffsqD_H0;
    scalar diffsqL_H1, diffsqR_H1, diffsqU_H1, diffsqD_H1;
    scalar diffsqL_H2, diffsqR_H2, diffsqU_H2, diffsqD_H2;
    scalar gradH_H0, gradV_H0;
    scalar gradH_H1, gradV_H1;
    scalar gradH_H2, gradV_H2;


    for(int x =  R+F_R; x < W - R - F_R; ++x) {
        int y = R+F_R;
        for(y; y < H -  R - F_R; y += 8) {
                // All operations have latency 4 and are executed on ports 0&1
                // Therefore we exectute the following:
                // Port 0: L0|L0²|U0|U0²|gradH0|L1|L1²|U1|U1²|gradH1|grad0
                // Port 1: R0|R0²|D0|D0²|gradV0|R1|R1²|D1|D1²|gradV1|grad1
                // We need to unroll 8 times to have correct scheduling:
                // A0A1 | C2D0 | F1F2
                //  A2B0 | D1D2 | G0G1
                //   B1B2 | E0E1 | G2H0
                //    C0C1 | E2F0 | H1H2

                // A0A1
                diffL_A0 = features[3 * (x * W + y) + 0] - features[3 * ((x - 1) * W + y) + 0];
                diffR_A0 = features[3 * (x * W + y) + 0] - features[3 * ((x + 1) * W + y) + 0];
                diffsqL_A0 = diffL_A0 * diffL_A0;
                diffsqR_A0 = diffR_A0 * diffR_A0;
                diffU_A0 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y - 1) + 0];
                diffD_A0 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y + 1) + 0];
                diffsqU_A0 = diffU_A0 * diffU_A0;
                diffsqD_A0 = diffD_A0 * diffD_A0;
                gradH_A0 = fmin(diffsqL_A0, diffsqR_A0);
                gradV_A0 = fmin(diffsqU_A0, diffsqD_A0);

                diffL_A1 = features[3 * (x * W + y) + 1] - features[3 * ((x - 1) * W + y) + 1];
                diffR_A1 = features[3 * (x * W + y) + 1] - features[3 * ((x + 1) * W + y) + 1];
                diffsqL_A1 = diffL_A1 * diffL_A1;
                diffsqR_A1 = diffR_A1 * diffR_A1;
                diffU_A1 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y - 1) + 1];
                diffD_A1 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y + 1) + 1];
                diffsqU_A1 = diffU_A1 * diffU_A1;
                diffsqD_A1 = diffD_A1 * diffD_A1;
                gradH_A1 = fmin(diffsqL_A1, diffsqR_A1);
                gradV_A1 = fmin(diffsqU_A1, diffsqD_A1);

                gradients[3 * (x * W + y) + 0] = gradH_A0 + gradV_A0;
                gradients[3 * (x * W + y) + 1] = gradH_A1 + gradV_A1;

                // A2B0
                diffL_A2 = features[3 * (x * W + y) + 2] - features[3 * ((x - 1) * W + y) + 2];
                diffR_A2 = features[3 * (x * W + y) + 2] - features[3 * ((x + 1) * W + y) + 2];
                diffsqL_A2 = diffL_A2 * diffL_A2;
                diffsqR_A2 = diffR_A2 * diffR_A2;
                diffU_A2 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y - 1) + 2];
                diffD_A2 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y + 1) + 2];
                diffsqU_A2 = diffU_A2 * diffU_A2;
                diffsqD_A2 = diffD_A2 * diffD_A2;
                gradH_A2 = fmin(diffsqL_A2, diffsqR_A2);
                gradV_A2 = fmin(diffsqU_A2, diffsqD_A2);

                diffL_B0 = features[3 * (x * W + y + 1) + 0] - features[3 * ((x - 0) * W + y + 1) + 0];
                diffR_B0 = features[3 * (x * W + y + 1) + 0] - features[3 * ((x + 0) * W + y + 1) + 0];
                diffsqL_B0 = diffL_B0 * diffL_B0;
                diffsqR_B0 = diffR_B0 * diffR_B0;
                diffU_B0 = features[3 * (x * W + y + 1) + 0] - features[3 * (x * W + y + 1 - 0) + 0];
                diffD_B0 = features[3 * (x * W + y + 1) + 0] - features[3 * (x * W + y + 1 + 0) + 0];
                diffsqU_B0 = diffU_B0 * diffU_B0;
                diffsqD_B0 = diffD_B0 * diffD_B0;
                gradH_B0 = fmin(diffsqL_B0, diffsqR_B0);
                gradV_B0 = fmin(diffsqU_B0, diffsqD_B0);

                gradients[3 * (x * W + y) + 0] = gradH_A2 + gradV_A2;
                gradients[3 * (x * W + y + 1) + 0] = gradH_B0 + gradV_B0;

                // B1B2
                diffL_B1 = features[3 * (x * W + y + 1) + 1] - features[3 * ((x - 1) * W + y + 1) + 1];
                diffR_B1 = features[3 * (x * W + y + 1) + 1] - features[3 * ((x + 1) * W + y + 1) + 1];
                diffsqL_B1 = diffL_B1 * diffL_B1;
                diffsqR_B1 = diffR_B1 * diffR_B1;
                diffU_B1 = features[3 * (x * W + y + 1) + 1] - features[3 * (x * W + y + 1 - 1) + 1];
                diffD_B1 = features[3 * (x * W + y + 1) + 1] - features[3 * (x * W + y + 1 + 1) + 1];
                diffsqU_B1 = diffU_B1 * diffU_B1;
                diffsqD_B1 = diffD_B1 * diffD_B1;
                gradH_B1 = fmin(diffsqL_B1, diffsqR_B1);
                gradV_B1 = fmin(diffsqU_B1, diffsqD_B1);

                diffL_B2 = features[3 * (x * W + y + 1) + 2] - features[3 * ((x - 1) * W + y + 1) + 2];
                diffR_B2 = features[3 * (x * W + y + 1) + 2] - features[3 * ((x + 1) * W + y + 1) + 2];
                diffsqL_B2 = diffL_B2 * diffL_B2;
                diffsqR_B2 = diffR_B2 * diffR_B2;
                diffU_B2 = features[3 * (x * W + y + 1) + 2] - features[3 * (x * W + y + 1 - 1) + 2];
                diffD_B2 = features[3 * (x * W + y + 1) + 2] - features[3 * (x * W + y + 1 + 1) + 2];
                diffsqU_B2 = diffU_B2 * diffU_B2;
                diffsqD_B2 = diffD_B2 * diffD_B2;
                gradH_B2 = fmin(diffsqL_B2, diffsqR_B2);
                gradV_B2 = fmin(diffsqU_B2, diffsqD_B2);

                gradients[3 * (x * W + y + 1) + 1] = gradH_B1 + gradV_B1;
                gradients[3 * (x * W + y + 1) + 2] = gradH_B2 + gradV_B2;

                // C0C1
                diffL_C0 = features[3 * (x * W + y + 2) + 0] - features[3 * ((x - 1) * W + y + 2) + 0];
                diffR_C0 = features[3 * (x * W + y + 2) + 0] - features[3 * ((x + 1) * W + y + 2) + 0];
                diffsqL_C0 = diffL_C0 * diffL_C0;
                diffsqR_C0 = diffR_C0 * diffR_C0;
                diffU_C0 = features[3 * (x * W + y + 2) + 0] - features[3 * (x * W + y + 2 - 1) + 0];
                diffD_C0 = features[3 * (x * W + y + 2) + 0] - features[3 * (x * W + y + 2 + 1) + 0];
                diffsqU_C0 = diffU_C0 * diffU_C0;
                diffsqD_C0 = diffD_C0 * diffD_C0;
                gradH_C0 = fmin(diffsqL_C0, diffsqR_C0);
                gradV_C0 = fmin(diffsqU_C0, diffsqD_C0);

                diffL_C1 = features[3 * (x * W + y + 2) + 1] - features[3 * ((x - 1) * W + y + 2) + 1];
                diffR_C1 = features[3 * (x * W + y + 2) + 1] - features[3 * ((x + 1) * W + y + 2) + 1];
                diffsqL_C1 = diffL_C1 * diffL_C1;
                diffsqR_C1 = diffR_C1 * diffR_C1;
                diffU_C1 = features[3 * (x * W + y + 2) + 1] - features[3 * (x * W + y + 2 - 1) + 1];
                diffD_C1 = features[3 * (x * W + y + 2) + 1] - features[3 * (x * W + y + 2 + 1) + 1];
                diffsqU_C1 = diffU_C1 * diffU_C1;
                diffsqD_C1 = diffD_C1 * diffD_C1;
                gradH_C1 = fmin(diffsqL_C1, diffsqR_C1);
                gradV_C1 = fmin(diffsqU_C1, diffsqD_C1);

                gradients[3 * (x * W + y + 2) + 0] = gradH_C0 + gradV_C0;
                gradients[3 * (x * W + y + 2) + 1] = gradH_C1 + gradV_C1;

                // C2D0
                diffL_C2 = features[3 * (x * W + y + 2) + 2] - features[3 * ((x - 1) * W + y + 2) + 2];
                diffR_C2 = features[3 * (x * W + y + 2) + 2] - features[3 * ((x + 1) * W + y + 2) + 2];
                diffsqL_C2 = diffL_C2 * diffL_C2;
                diffsqR_C2 = diffR_C2 * diffR_C2;
                diffU_C2 = features[3 * (x * W + y + 2) + 2] - features[3 * (x * W + y + 2 - 1) + 2];
                diffD_C2 = features[3 * (x * W + y + 2) + 2] - features[3 * (x * W + y + 2 + 1) + 2];
                diffsqU_C2 = diffU_C2 * diffU_C2;
                diffsqD_C2 = diffD_C2 * diffD_C2;
                gradH_C2 = fmin(diffsqL_C2, diffsqR_C2);
                gradV_C2 = fmin(diffsqU_C2, diffsqD_C2);

                diffL_D0 = features[3 * (x * W + y + 3) + 0] - features[3 * ((x - 0) * W + y + 3) + 0];
                diffR_D0 = features[3 * (x * W + y + 3) + 0] - features[3 * ((x + 0) * W + y + 3) + 0];
                diffsqL_D0 = diffL_D0 * diffL_D0;
                diffsqR_D0 = diffR_D0 * diffR_D0;
                diffU_D0 = features[3 * (x * W + y + 3) + 0] - features[3 * (x * W + y + 3 - 0) + 0];
                diffD_D0 = features[3 * (x * W + y + 3) + 0] - features[3 * (x * W + y + 3 + 0) + 0];
                diffsqU_D0 = diffU_D0 * diffU_D0;
                diffsqD_D0 = diffD_D0 * diffD_D0;
                gradH_D0 = fmin(diffsqL_D0, diffsqR_D0);
                gradV_D0 = fmin(diffsqU_D0, diffsqD_D0);

                gradients[3 * (x * W + y + 2) + 0] = gradH_C2 + gradV_C2;
                gradients[3 * (x * W + y + 3) + 0] = gradH_D0 + gradV_D0;

                // D1D2
                diffL_D1 = features[3 * (x * W + y + 3) + 1] - features[3 * ((x - 1) * W + y + 3) + 1];
                diffR_D1 = features[3 * (x * W + y + 3) + 1] - features[3 * ((x + 1) * W + y + 3) + 1];
                diffsqL_D1 = diffL_D1 * diffL_D1;
                diffsqR_D1 = diffR_D1 * diffR_D1;
                diffU_D1 = features[3 * (x * W + y + 3) + 1] - features[3 * (x * W + y + 3 - 1) + 1];
                diffD_D1 = features[3 * (x * W + y + 3) + 1] - features[3 * (x * W + y + 3 + 1) + 1];
                diffsqU_D1 = diffU_D1 * diffU_D1;
                diffsqD_D1 = diffD_D1 * diffD_D1;
                gradH_D1 = fmin(diffsqL_D1, diffsqR_D1);
                gradV_D1 = fmin(diffsqU_D1, diffsqD_D1);

                diffL_D2 = features[3 * (x * W + y + 3) + 2] - features[3 * ((x - 1) * W + y + 3) + 2];
                diffR_D2 = features[3 * (x * W + y + 3) + 2] - features[3 * ((x + 1) * W + y + 3) + 2];
                diffsqL_D2 = diffL_D2 * diffL_D2;
                diffsqR_D2 = diffR_D2 * diffR_D2;
                diffU_D2 = features[3 * (x * W + y + 3) + 2] - features[3 * (x * W + y + 3 - 1) + 2];
                diffD_D2 = features[3 * (x * W + y + 3) + 2] - features[3 * (x * W + y + 3 + 1) + 2];
                diffsqU_D2 = diffU_D2 * diffU_D2;
                diffsqD_D2 = diffD_D2 * diffD_D2;
                gradH_D2 = fmin(diffsqL_D2, diffsqR_D2);
                gradV_D2 = fmin(diffsqU_D2, diffsqD_D2);

                gradients[3 * (x * W + y + 3) + 1] = gradH_D1 + gradV_D1;
                gradients[3 * (x * W + y + 3) + 2] = gradH_D2 + gradV_D2;

                // E0E1
                diffL_E0 = features[3 * (x * W + y + 4) + 0] - features[3 * ((x - 1) * W + y + 4) + 0];
                diffR_E0 = features[3 * (x * W + y + 4) + 0] - features[3 * ((x + 1) * W + y + 4) + 0];
                diffsqL_E0 = diffL_E0 * diffL_E0;
                diffsqR_E0 = diffR_E0 * diffR_E0;
                diffU_E0 = features[3 * (x * W + y + 4) + 0] - features[3 * (x * W + y + 4 - 1) + 0];
                diffD_E0 = features[3 * (x * W + y + 4) + 0] - features[3 * (x * W + y + 4 + 1) + 0];
                diffsqU_E0 = diffU_E0 * diffU_E0;
                diffsqD_E0 = diffD_E0 * diffD_E0;
                gradH_E0 = fmin(diffsqL_E0, diffsqR_E0);
                gradV_E0 = fmin(diffsqU_E0, diffsqD_E0);

                diffL_E1 = features[3 * (x * W + y + 4) + 1] - features[3 * ((x - 1) * W + y + 4) + 1];
                diffR_E1 = features[3 * (x * W + y + 4) + 1] - features[3 * ((x + 1) * W + y + 4) + 1];
                diffsqL_E1 = diffL_E1 * diffL_E1;
                diffsqR_E1 = diffR_E1 * diffR_E1;
                diffU_E1 = features[3 * (x * W + y + 4) + 1] - features[3 * (x * W + y + 4 - 1) + 1];
                diffD_E1 = features[3 * (x * W + y + 4) + 1] - features[3 * (x * W + y + 4 + 1) + 1];
                diffsqU_E1 = diffU_E1 * diffU_E1;
                diffsqD_E1 = diffD_E1 * diffD_E1;
                gradH_E1 = fmin(diffsqL_E1, diffsqR_E1);
                gradV_E1 = fmin(diffsqU_E1, diffsqD_E1);

                gradients[3 * (x * W + y + 4) + 0] = gradH_E0 + gradV_E0;
                gradients[3 * (x * W + y + 4) + 1] = gradH_E1 + gradV_E1;

                // E2F0
                diffL_E2 = features[3 * (x * W + y + 4) + 2] - features[3 * ((x - 1) * W + y + 4) + 2];
                diffR_E2 = features[3 * (x * W + y + 4) + 2] - features[3 * ((x + 1) * W + y + 4) + 2];
                diffsqL_E2 = diffL_E2 * diffL_E2;
                diffsqR_E2 = diffR_E2 * diffR_E2;
                diffU_E2 = features[3 * (x * W + y + 4) + 2] - features[3 * (x * W + y + 4 - 1) + 2];
                diffD_E2 = features[3 * (x * W + y + 4) + 2] - features[3 * (x * W + y + 4 + 1) + 2];
                diffsqU_E2 = diffU_E2 * diffU_E2;
                diffsqD_E2 = diffD_E2 * diffD_E2;
                gradH_E2 = fmin(diffsqL_E2, diffsqR_E2);
                gradV_E2 = fmin(diffsqU_E2, diffsqD_E2);

                diffL_F0 = features[3 * (x * W + y + 5) + 0] - features[3 * ((x - 0) * W + y + 5) + 0];
                diffR_F0 = features[3 * (x * W + y + 5) + 0] - features[3 * ((x + 0) * W + y + 5) + 0];
                diffsqL_F0 = diffL_F0 * diffL_F0;
                diffsqR_F0 = diffR_F0 * diffR_F0;
                diffU_F0 = features[3 * (x * W + y + 5) + 0] - features[3 * (x * W + y + 5 - 0) + 0];
                diffD_F0 = features[3 * (x * W + y + 5) + 0] - features[3 * (x * W + y + 5 + 0) + 0];
                diffsqU_F0 = diffU_F0 * diffU_F0;
                diffsqD_F0 = diffD_F0 * diffD_F0;
                gradH_F0 = fmin(diffsqL_F0, diffsqR_F0);
                gradV_F0 = fmin(diffsqU_F0, diffsqD_F0);

                gradients[3 * (x * W + y + 4) + 0] = gradH_E2 + gradV_E2;
                gradients[3 * (x * W + y + 5) + 0] = gradH_F0 + gradV_F0;

                // F1F2
                diffL_F1 = features[3 * (x * W + y + 5) + 1] - features[3 * ((x - 1) * W + y + 5) + 1];
                diffR_F1 = features[3 * (x * W + y + 5) + 1] - features[3 * ((x + 1) * W + y + 5) + 1];
                diffsqL_F1 = diffL_F1 * diffL_F1;
                diffsqR_F1 = diffR_F1 * diffR_F1;
                diffU_F1 = features[3 * (x * W + y + 5) + 1] - features[3 * (x * W + y + 5 - 1) + 1];
                diffD_F1 = features[3 * (x * W + y + 5) + 1] - features[3 * (x * W + y + 5 + 1) + 1];
                diffsqU_F1 = diffU_F1 * diffU_F1;
                diffsqD_F1 = diffD_F1 * diffD_F1;
                gradH_F1 = fmin(diffsqL_F1, diffsqR_F1);
                gradV_F1 = fmin(diffsqU_F1, diffsqD_F1);

                diffL_F2 = features[3 * (x * W + y + 5) + 2] - features[3 * ((x - 1) * W + y + 5) + 2];
                diffR_F2 = features[3 * (x * W + y + 5) + 2] - features[3 * ((x + 1) * W + y + 5) + 2];
                diffsqL_F2 = diffL_F2 * diffL_F2;
                diffsqR_F2 = diffR_F2 * diffR_F2;
                diffU_F2 = features[3 * (x * W + y + 5) + 2] - features[3 * (x * W + y + 5 - 1) + 2];
                diffD_F2 = features[3 * (x * W + y + 5) + 2] - features[3 * (x * W + y + 5 + 1) + 2];
                diffsqU_F2 = diffU_F2 * diffU_F2;
                diffsqD_F2 = diffD_F2 * diffD_F2;
                gradH_F2 = fmin(diffsqL_F2, diffsqR_F2);
                gradV_F2 = fmin(diffsqU_F2, diffsqD_F2);

                gradients[3 * (x * W + y + 5) + 1] = gradH_F1 + gradV_F1;
                gradients[3 * (x * W + y + 5) + 2] = gradH_F2 + gradV_F2;

                // G0G1
                diffL_G0 = features[3 * (x * W + y + 6) + 0] - features[3 * ((x - 1) * W + y + 6) + 0];
                diffR_G0 = features[3 * (x * W + y + 6) + 0] - features[3 * ((x + 1) * W + y + 6) + 0];
                diffsqL_G0 = diffL_G0 * diffL_G0;
                diffsqR_G0 = diffR_G0 * diffR_G0;
                diffU_G0 = features[3 * (x * W + y + 6) + 0] - features[3 * (x * W + y + 6 - 1) + 0];
                diffD_G0 = features[3 * (x * W + y + 6) + 0] - features[3 * (x * W + y + 6 + 1) + 0];
                diffsqU_G0 = diffU_G0 * diffU_G0;
                diffsqD_G0 = diffD_G0 * diffD_G0;
                gradH_G0 = fmin(diffsqL_G0, diffsqR_G0);
                gradV_G0 = fmin(diffsqU_G0, diffsqD_G0);

                diffL_G1 = features[3 * (x * W + y + 6) + 1] - features[3 * ((x - 1) * W + y + 6) + 1];
                diffR_G1 = features[3 * (x * W + y + 6) + 1] - features[3 * ((x + 1) * W + y + 6) + 1];
                diffsqL_G1 = diffL_G1 * diffL_G1;
                diffsqR_G1 = diffR_G1 * diffR_G1;
                diffU_G1 = features[3 * (x * W + y + 6) + 1] - features[3 * (x * W + y + 6 - 1) + 1];
                diffD_G1 = features[3 * (x * W + y + 6) + 1] - features[3 * (x * W + y + 6 + 1) + 1];
                diffsqU_G1 = diffU_G1 * diffU_G1;
                diffsqD_G1 = diffD_G1 * diffD_G1;
                gradH_G1 = fmin(diffsqL_G1, diffsqR_G1);
                gradV_G1 = fmin(diffsqU_G1, diffsqD_G1);

                gradients[3 * (x * W + y + 6) + 0] = gradH_G0 + gradV_G0;
                gradients[3 * (x * W + y + 6) + 1] = gradH_G1 + gradV_G1;

                // G2H0
                diffL_G2 = features[3 * (x * W + y + 6) + 2] - features[3 * ((x - 1) * W + y + 6) + 2];
                diffR_G2 = features[3 * (x * W + y + 6) + 2] - features[3 * ((x + 1) * W + y + 6) + 2];
                diffsqL_G2 = diffL_G2 * diffL_G2;
                diffsqR_G2 = diffR_G2 * diffR_G2;
                diffU_G2 = features[3 * (x * W + y + 6) + 2] - features[3 * (x * W + y + 6 - 1) + 2];
                diffD_G2 = features[3 * (x * W + y + 6) + 2] - features[3 * (x * W + y + 6 + 1) + 2];
                diffsqU_G2 = diffU_G2 * diffU_G2;
                diffsqD_G2 = diffD_G2 * diffD_G2;
                gradH_G2 = fmin(diffsqL_G2, diffsqR_G2);
                gradV_G2 = fmin(diffsqU_G2, diffsqD_G2);

                diffL_H0 = features[3 * (x * W + y + 7) + 0] - features[3 * ((x - 0) * W + y + 7) + 0];
                diffR_H0 = features[3 * (x * W + y + 7) + 0] - features[3 * ((x + 0) * W + y + 7) + 0];
                diffsqL_H0 = diffL_H0 * diffL_H0;
                diffsqR_H0 = diffR_H0 * diffR_H0;
                diffU_H0 = features[3 * (x * W + y + 7) + 0] - features[3 * (x * W + y + 7 - 0) + 0];
                diffD_H0 = features[3 * (x * W + y + 7) + 0] - features[3 * (x * W + y + 7 + 0) + 0];
                diffsqU_H0 = diffU_H0 * diffU_H0;
                diffsqD_H0 = diffD_H0 * diffD_H0;
                gradH_H0 = fmin(diffsqL_H0, diffsqR_H0);
                gradV_H0 = fmin(diffsqU_H0, diffsqD_H0);

                gradients[3 * (x * W + y + 6) + 0] = gradH_G2 + gradV_G2;
                gradients[3 * (x * W + y + 7) + 0] = gradH_H0 + gradV_H0;

                // H1H2
                diffL_H1 = features[3 * (x * W + y + 7) + 1] - features[3 * ((x - 1) * W + y + 7) + 1];
                diffR_H1 = features[3 * (x * W + y + 7) + 1] - features[3 * ((x + 1) * W + y + 7) + 1];
                diffsqL_H1 = diffL_H1 * diffL_H1;
                diffsqR_H1 = diffR_H1 * diffR_H1;
                diffU_H1 = features[3 * (x * W + y + 7) + 1] - features[3 * (x * W + y + 7 - 1) + 1];
                diffD_H1 = features[3 * (x * W + y + 7) + 1] - features[3 * (x * W + y + 7 + 1) + 1];
                diffsqU_H1 = diffU_H1 * diffU_H1;
                diffsqD_H1 = diffD_H1 * diffD_H1;
                gradH_H1 = fmin(diffsqL_H1, diffsqR_H1);
                gradV_H1 = fmin(diffsqU_H1, diffsqD_H1);

                diffL_H2 = features[3 * (x * W + y + 7) + 2] - features[3 * ((x - 1) * W + y + 7) + 2];
                diffR_H2 = features[3 * (x * W + y + 7) + 2] - features[3 * ((x + 1) * W + y + 7) + 2];
                diffsqL_H2 = diffL_H2 * diffL_H2;
                diffsqR_H2 = diffR_H2 * diffR_H2;
                diffU_H2 = features[3 * (x * W + y + 7) + 2] - features[3 * (x * W + y + 7 - 1) + 2];
                diffD_H2 = features[3 * (x * W + y + 7) + 2] - features[3 * (x * W + y + 7 + 1) + 2];
                diffsqU_H2 = diffU_H2 * diffU_H2;
                diffsqD_H2 = diffD_H2 * diffD_H2;
                gradH_H2 = fmin(diffsqL_H2, diffsqR_H2);
                gradV_H2 = fmin(diffsqU_H2, diffsqD_H2);

                gradients[3 * (x * W + y + 7) + 1] = gradH_H1 + gradV_H1;
                gradients[3 * (x * W + y + 7) + 2] = gradH_H2 + gradV_H2;
                
        } 

        // Remaining
        if(y > H -  R - F_R)
            y -= 8;

        for(y; y < H -  R - F_R; ++y) {  
            scalar diffL_00 = features[3 * (x * W + y) + 0] - features[3 * ((x - 1) * W + y) + 0];
            scalar diffL_01 = features[3 * (x * W + y) + 1] - features[3 * ((x - 1) * W + y) + 1];
            scalar diffL_02 = features[3 * (x * W + y) + 2] - features[3 * ((x - 1) * W + y) + 2];

            scalar diffR_00 = features[3 * (x * W + y) + 0] - features[3 * ((x + 1) * W + y) + 0];
            scalar diffR_01 = features[3 * (x * W + y) + 1] - features[3 * ((x + 1) * W + y) + 1];
            scalar diffR_02 = features[3 * (x * W + y) + 2] - features[3 * ((x + 1) * W + y) + 2];

            scalar diffU_00 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y - 1) + 0];
            scalar diffU_01 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y - 1) + 1];
            scalar diffU_02 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y - 1) + 2];

            scalar diffD_00 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y + 1) + 0];
            scalar diffD_01 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y + 1) + 1];
            scalar diffD_02 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y + 1) + 2];

            gradients[3 * (x * W + y) + 0] = fmin(diffL_01*diffL_01, diffR_00*diffR_00) + fmin(diffU_00*diffU_00, diffD_00*diffD_00);
            gradients[3 * (x * W + y) + 1] = fmin(diffL_01*diffL_01, diffR_01*diffR_01) + fmin(diffU_01*diffU_01, diffD_01*diffD_01);
            gradients[3 * (x * W + y) + 2] = fmin(diffL_02*diffL_02, diffR_02*diffR_02) + fmin(diffU_02*diffU_02, diffD_02*diffD_02);
        } 

    }
}

void color_weights_SSA(scalar *temp, scalar *color, scalar *color_var, const int r_x, const int r_y, const int W, const int H, const int R) {
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sqdist_00 = color[3 * (xp * W + yp) + 0] - color[3 * (xq * W + yq) + 0];
                    scalar sqdist_01 = color[3 * (xp * W + yp) + 1] - color[3 * (xq * W + yq) + 1];
                    scalar sqdist_02 = color[3 * (xp * W + yp) + 2] - color[3 * (xq * W + yq) + 2];

                    sqdist_00 *= sqdist_00;
                    sqdist_01 *= sqdist_01;
                    sqdist_02 *= sqdist_02;

                    scalar var_cancel_00 = color_var[3 * (xp * W + yp) + 0] + fmin(color_var[3 * (xp * W + yp) + 0], color_var[3 * (xq * W + yq) + 0]);
                    scalar var_cancel_01 = color_var[3 * (xp * W + yp) + 1] + fmin(color_var[3 * (xp * W + yp) + 1], color_var[3 * (xq * W + yq) + 1]);
                    scalar var_cancel_02 = color_var[3 * (xp * W + yp) + 2] + fmin(color_var[3 * (xp * W + yp) + 2], color_var[3 * (xq * W + yq) + 2]);

                    scalar var_term_00 = color_var[3 * (xp * W + yp) + 0] + color_var[3 * (xq * W + yq) + 0];
                    scalar var_term_01 = color_var[3 * (xp * W + yp) + 1] + color_var[3 * (xq * W + yq) + 1];
                    scalar var_term_02 = color_var[3 * (xp * W + yp) + 2] + color_var[3 * (xq * W + yq) + 2];

                    scalar normalization_r_00 = EPSILON + KC_SQUARED_R*(var_term_00);
                    scalar normalization_r_01 = EPSILON + KC_SQUARED_R*(var_term_01);
                    scalar normalization_r_02 = EPSILON + KC_SQUARED_R*(var_term_02);

                    scalar dist_var_00 = sqdist_00 - var_cancel_00;
                    scalar dist_var_01 = sqdist_01 - var_cancel_01;
                    scalar dist_var_02 = sqdist_02 - var_cancel_02;

                    temp[xp * W + yp] = ((dist_var_00) / normalization_r_00) + ((dist_var_01) / normalization_r_01) + ((dist_var_02) / normalization_r_02);
                }
            }
}

void precompute_features_SSA(scalar *features_weights_r_num, scalar *features_weights_r_den, scalar *features_weights_b, scalar *features, scalar *features_var, scalar *gradients, const int r_x, const int r_y, const int R, const int W, const int H) {
            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r_0_num = 0.f;
                    scalar df_r_1_num = 0.f;
                    scalar df_r_2_num = 0.f;
                    scalar df_r_3_num = 0.f;
                    scalar df_r_0_den = 1.f;
                    scalar df_r_1_den = 1.f;
                    scalar df_r_2_den = 1.f;
                    scalar df_r_3_den = 1.f;

                    scalar df_b_0_num = 0.f;
                    scalar df_b_1_num = 0.f;
                    scalar df_b_2_num = 0.f;
                    scalar df_b_3_num = 0.f;
                    scalar df_b_0_den = 1.f;
                    scalar df_b_1_den = 1.f;
                    scalar df_b_2_den = 1.f;
                    scalar df_b_3_den = 1.f;

                    for(int j=0; j<NB_FEATURES;++j){
                        
                        scalar sqdist_0 = features[3 * (xp * W + yp) + j] - features[3 * (xq * W + yq) + j];
                        scalar sqdist_1 = features[3 * (xp * W + yp+1) + j] - features[3 * (xq * W + yq+1) + j];
                        scalar sqdist_2 = features[3 * (xp * W + yp+2) + j] - features[3 * (xq * W + yq+2) + j];
                        scalar sqdist_3 = features[3 * (xp * W + yp+3) + j] - features[3 * (xq * W + yq+3) + j];

                        sqdist_0 *= sqdist_0;
                        sqdist_1 *= sqdist_1;
                        sqdist_2 *= sqdist_2;
                        sqdist_3 *= sqdist_3;

                        scalar var_cancel_0 = features_var[3 * (xp * W + yp) + j] + fmin(features_var[3 * (xp * W + yp) + j], features_var[3 * (xq * W + yq) + j]);
                        scalar var_cancel_1 = features_var[3 * (xp * W + yp+1) + j] + fmin(features_var[3 * (xp * W + yp+1) + j], features_var[3 * (xq * W + yq+1) + j]);
                        scalar var_cancel_2 = features_var[3 * (xp * W + yp+2) + j] + fmin(features_var[3 * (xp * W + yp+2) + j], features_var[3 * (xq * W + yq+2) + j]);
                        scalar var_cancel_3 = features_var[3 * (xp * W + yp+3) + j] + fmin(features_var[3 * (xp * W + yp+3) + j], features_var[3 * (xq * W + yq+3) + j]);
                        
                        scalar var_max_0 = fmax(features_var[3 * (xp * W + yp) + j], gradients[3 * (xp * W + yp) + j]);
                        scalar var_max_1 = fmax(features_var[3 * (xp * W + yp+1) + j], gradients[3 * (xp * W + yp+1) + j]);
                        scalar var_max_2 = fmax(features_var[3 * (xp * W + yp+2) + j], gradients[3 * (xp * W + yp+2) + j]);
                        scalar var_max_3 = fmax(features_var[3 * (xp * W + yp+3) + j], gradients[3 * (xp * W + yp+3) + j]);

                        scalar normalization_r_0 = KF_SQUARED_R*fmax(TAU_R, var_max_0);
                        scalar normalization_r_1 = KF_SQUARED_R*fmax(TAU_R, var_max_1);
                        scalar normalization_r_2 = KF_SQUARED_R*fmax(TAU_R, var_max_2);
                        scalar normalization_r_3 = KF_SQUARED_R*fmax(TAU_R, var_max_3);

                        scalar normalization_b_0 = KF_SQUARED_B*fmax(TAU_B, var_max_0);
                        scalar normalization_b_1 = KF_SQUARED_B*fmax(TAU_B, var_max_1);
                        scalar normalization_b_2 = KF_SQUARED_B*fmax(TAU_B, var_max_2);
                        scalar normalization_b_3 = KF_SQUARED_B*fmax(TAU_B, var_max_3);

                        scalar dist_var_0 = sqdist_0 - var_cancel_0;
                        scalar dist_var_1 = sqdist_1 - var_cancel_1;
                        scalar dist_var_2 = sqdist_2 - var_cancel_2;
                        scalar dist_var_3 = sqdist_3 - var_cancel_3;

                        if(df_r_0_num * normalization_r_0 <= df_r_0_den * dist_var_0) {
                            df_r_0_num = dist_var_0;
                            df_r_0_den = normalization_r_0;
                        }
                        if(df_r_1_num * normalization_r_1 <= df_r_1_den * dist_var_1) {
                            df_r_1_num = dist_var_1;
                            df_r_1_den = normalization_r_1;
                        }
                        if(df_r_2_num * normalization_r_2 <= df_r_2_den * dist_var_2) {
                            df_r_2_num = dist_var_2;
                            df_r_2_den = normalization_r_2;
                        }
                        if(df_r_3_num * normalization_r_3 <= df_r_3_den * dist_var_3) {
                            df_r_3_num = dist_var_3;
                            df_r_3_den = normalization_r_3;
                        }

                        if(df_b_0_num * normalization_b_0 <= df_b_0_den * dist_var_0) {
                            df_b_0_num = dist_var_0;
                            df_b_0_den = normalization_b_0;
                        }
                        if(df_b_1_num * normalization_b_1 <= df_b_1_den * dist_var_1) {
                            df_b_1_num = dist_var_1;
                            df_b_1_den = normalization_b_1;
                        }
                        if(df_b_2_num * normalization_b_2 <= df_b_2_den * dist_var_2) {
                            df_b_2_num = dist_var_2;
                            df_b_2_den = normalization_b_2;
                        }
                        if(df_b_3_num * normalization_b_3 <= df_b_3_den * dist_var_3) {
                            df_b_3_num = dist_var_3;
                            df_b_3_den = normalization_b_3;
                        }
                    }

                    features_weights_r_num[xp * W + yp] = df_r_0_num;
                    features_weights_r_den[xp * W + yp] = df_r_0_den;
                    features_weights_r_num[xp * W + yp+1] = df_r_1_num;
                    features_weights_r_den[xp * W + yp+1] = df_r_1_den;
                    features_weights_r_num[xp * W + yp+2] = df_r_2_num;
                    features_weights_r_den[xp * W + yp+2] = df_r_2_den;
                    features_weights_r_num[xp * W + yp+3] = df_r_3_num;
                    features_weights_r_den[xp * W + yp+3] = df_r_3_den;
                    features_weights_b[xp * W + yp] = exp(-df_b_0_num / df_b_0_den);
                    features_weights_b[xp * W + yp+1] = exp(-df_b_0_num / df_b_0_den);
                    features_weights_b[xp * W + yp+2] = exp(-df_b_0_num / df_b_0_den);
                    features_weights_b[xp * W + yp+3] = exp(-df_b_0_num / df_b_0_den);
                } 
            }
}

void candidate_R_SSA(scalar *output_r, scalar *weight_sum, scalar *temp, scalar *temp2_r, scalar *features_weights_r_num, scalar *features_weights_r_den, scalar *color, const int r_x, const int r_y, const int neigh_r, const int R, const int W, const int H) {
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=8) {
                    scalar sum_r0 = 0.f;
                    scalar sum_r1 = 0.f;
                    scalar sum_r2 = 0.f;
                    scalar sum_r3 = 0.f;
                    scalar sum_r4 = 0.f;
                    scalar sum_r5 = 0.f;
                    scalar sum_r6 = 0.f;
                    scalar sum_r7 = 0.f;
                     /*
                    scalar sum_r8 = 0.f;
                    scalar sum_r9 = 0.f;
                    scalar sum_r10 = 0.f;
                    scalar sum_r11 = 0.f;
                    scalar sum_r12 = 0.f;
                    scalar sum_r13 = 0.f;
                    scalar sum_r14 = 0.f;
                    scalar sum_r15 = 0.f;
                    */

                    for (int k=-F_R; k<=F_R; k++){
                        sum_r0 += temp[xp * W + yp+k];
                        sum_r1 += temp[xp * W + yp+k+1];
                        sum_r2 += temp[xp * W + yp+k+2];
                        sum_r3 += temp[xp * W + yp+k+3];
                        sum_r4 += temp[xp * W + yp+k+4];
                        sum_r5 += temp[xp * W + yp+k+5];
                        sum_r6 += temp[xp * W + yp+k+6];
                        sum_r7 += temp[xp * W + yp+k+7];
                        /*
                        sum_r8 += temp[xp * W + yp+k+8];
                        sum_r9 += temp[xp * W + yp+k+9];
                        sum_r10 += temp[xp * W + yp+k+10];
                        sum_r11 += temp[xp * W + yp+k+11];
                        sum_r12 += temp[xp * W + yp+k+12];
                        sum_r13 += temp[xp * W + yp+k+13];
                        sum_r14 += temp[xp * W + yp+k+14];
                        sum_r15 += temp[xp * W + yp+k+15];
                        */
                    }
                    temp2_r[xp * W + yp] = sum_r0;
                    temp2_r[xp * W + yp+1] = sum_r1;
                    temp2_r[xp * W + yp+2] = sum_r2;
                    temp2_r[xp * W + yp+3] = sum_r3;
                    temp2_r[xp * W + yp+4] = sum_r4;
                    temp2_r[xp * W + yp+5] = sum_r5;
                    temp2_r[xp * W + yp+6] = sum_r6;
                    temp2_r[xp * W + yp+7] = sum_r7;
                    /*
                    temp2_r[xp * W + yp+8] = sum_r8;
                    temp2_r[xp * W + yp+9] = sum_r9;
                    temp2_r[xp * W + yp+10] = sum_r10;
                    temp2_r[xp * W + yp+11] = sum_r11;
                    temp2_r[xp * W + yp+12] = sum_r12;
                    temp2_r[xp * W + yp+13] = sum_r13;
                    temp2_r[xp * W + yp+14] = sum_r14;
                    temp2_r[xp * W + yp+15] = sum_r15;
                    */
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
                    /*
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;
                    scalar sum_8 = 0.f;
                    scalar sum_9 = 0.f;
                    scalar sum_10 = 0.f;
                    scalar sum_11 = 0.f;
                    scalar sum_12 = 0.f;
                    scalar sum_13 = 0.f;
                    scalar sum_14 = 0.f;
                    scalar sum_15 = 0.f;
                    */

                    for (int k=-F_R; k<=F_R; k++){
                        sum_0 += temp2_r[(xp+k)*W + yp];
                        sum_1 += temp2_r[(xp+k)*W + yp+1];
                        sum_2 += temp2_r[(xp+k)*W + yp+2];
                        sum_3 += temp2_r[(xp+k)*W + yp+3];
                        /*
                        sum_4 += temp2_r[(xp+k)*W + yp+4];
                        sum_5 += temp2_r[(xp+k)*W + yp+5];
                        sum_6 += temp2_r[(xp+k)*W + yp+6];
                        sum_7 += temp2_r[(xp+k)*W + yp+7];
                        sum_8 += temp2_r[(xp+k)*W + yp+8];
                        sum_9 += temp2_r[(xp+k)*W + yp+9];
                        sum_10 += temp2_r[(xp+k)*W + yp+10];
                        sum_11 += temp2_r[(xp+k)*W + yp+11];
                        sum_12 += temp2_r[(xp+k)*W + yp+12];
                        sum_13 += temp2_r[(xp+k)*W + yp+13];
                        sum_14 += temp2_r[(xp+k)*W + yp+14];
                        sum_15 += temp2_r[(xp+k)*W + yp+15];
                        */
                    }

                    scalar weight_0, weight_1, weight_2, weight_3;

                    scalar fweights_0_num = features_weights_r_num[xp * W + yp];
                    scalar fweights_0_den = features_weights_r_den[xp * W + yp];
                    scalar fweights_1_num = features_weights_r_num[xp * W + yp + 1];
                    scalar fweights_1_den = features_weights_r_den[xp * W + yp + 1];
                    scalar fweights_2_num = features_weights_r_num[xp * W + yp + 2];
                    scalar fweights_2_den = features_weights_r_den[xp * W + yp + 2];
                    scalar fweights_3_num = features_weights_r_num[xp * W + yp + 3];
                    scalar fweights_3_den = features_weights_r_den[xp * W + yp + 3];

                    if(sum_0 * fweights_0_den <= neigh_r * fweights_0_num)
                        weight_0 = exp(-fweights_0_num / fweights_0_den);
                    else
                        weight_0 = exp(-sum_0 / neigh_r);
                    if(sum_1 * fweights_1_den <= neigh_r * fweights_1_num)
                        weight_1 = exp(-fweights_1_num / fweights_1_den);
                    else
                        weight_1 = exp(-sum_1 / neigh_r);
                    if(sum_2 * fweights_2_den <= neigh_r * fweights_2_num)
                        weight_2 = exp(-fweights_2_num / fweights_2_den);
                    else
                        weight_2 = exp(-sum_2 / neigh_r);
                    if(sum_3 * fweights_3_den <= neigh_r * fweights_3_num)
                        weight_3 = exp(-fweights_3_num / fweights_3_den);
                    else
                        weight_3 = exp(-sum_3 / neigh_r);

                    
                    weight_sum[3 * (xp * W + yp)] += weight_0;
                    weight_sum[3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_r[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_r[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_r[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_r[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_r[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_r[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_r[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_r[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_r[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_r[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_r[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_r[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_r[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_r[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_r[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_r[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }
}

void candidate_G_SSA(scalar *output_g, scalar *weight_sum, scalar *temp, scalar *temp2_g, scalar *features_weights_r_num, scalar *features_weights_r_den, scalar *color, const int r_x, const int r_y, const int neigh_g, const int R, const int W, const int H) {
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=8) {
                    
                    scalar sum_g0 = 0.f;
                    scalar sum_g1 = 0.f;
                    scalar sum_g2 = 0.f;
                    scalar sum_g3 = 0.f;
                    scalar sum_g4 = 0.f;
                    scalar sum_g5 = 0.f;
                    scalar sum_g6 = 0.f;
                    scalar sum_g7 = 0.f;
                    /*
                    scalar sum_g8 = 0.f;
                    scalar sum_g9 = 0.f;
                    scalar sum_g10 = 0.f;
                    scalar sum_g11 = 0.f;
                    scalar sum_g12 = 0.f;
                    scalar sum_g13 = 0.f;
                    scalar sum_g14 = 0.f;
                    scalar sum_g15 = 0.f;
                    */

                    for (int k=-F_G; k<=F_G; k++){
                        sum_g0 += temp[xp * W + yp+k];
                        sum_g1 += temp[xp * W + yp+k+1];
                        sum_g2 += temp[xp * W + yp+k+2];
                        sum_g3 += temp[xp * W + yp+k+3];
                        sum_g4 += temp[xp * W + yp+k+4];
                        sum_g5 += temp[xp * W + yp+k+5];
                        sum_g6 += temp[xp * W + yp+k+6];
                        sum_g7 += temp[xp * W + yp+k+7];
                        /*
                        sum_g8 += temp[xp * W + yp+k+8];
                        sum_g9 += temp[xp * W + yp+k+9];
                        sum_g10 += temp[xp * W + yp+k+10];
                        sum_g11 += temp[xp * W + yp+k+11];
                        sum_g12 += temp[xp * W + yp+k+12];
                        sum_g13 += temp[xp * W + yp+k+13];
                        sum_g14 += temp[xp * W + yp+k+14];
                        sum_g15 += temp[xp * W + yp+k+15];
                        */
                    }
                    temp2_g[xp * W + yp] = sum_g0;
                    temp2_g[xp * W + yp+1] = sum_g1;
                    temp2_g[xp * W + yp+2] = sum_g2;
                    temp2_g[xp * W + yp+3] = sum_g3;
                    temp2_g[xp * W + yp+4] = sum_g4;
                    temp2_g[xp * W + yp+5] = sum_g5;
                    temp2_g[xp * W + yp+6] = sum_g6;
                    temp2_g[xp * W + yp+7] = sum_g7;
                    /*
                    temp2_g[xp * W + yp+8] = sum_g8;
                    temp2_g[xp * W + yp+9] = sum_g9;
                    temp2_g[xp * W + yp+10] = sum_g10;
                    temp2_g[xp * W + yp+11] = sum_g11;
                    temp2_g[xp * W + yp+12] = sum_g12;
                    temp2_g[xp * W + yp+13] = sum_g13;
                    temp2_g[xp * W + yp+14] = sum_g14;
                    temp2_g[xp * W + yp+15] = sum_g15;
                    */
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
                    /*
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;
                    scalar sum_8 = 0.f;
                    scalar sum_9 = 0.f;
                    scalar sum_10 = 0.f;
                    scalar sum_11 = 0.f;
                    scalar sum_12 = 0.f;
                    scalar sum_13 = 0.f;
                    scalar sum_14 = 0.f;
                    scalar sum_15 = 0.f;
                    */

                    for (int k=-F_G; k<=F_G; k++){
                        sum_0 += temp2_g[(xp+k)*W + yp];
                        sum_1 += temp2_g[(xp+k)*W + yp+1];
                        sum_2 += temp2_g[(xp+k)*W + yp+2];
                        sum_3 += temp2_g[(xp+k)*W + yp+3];
                        /*
                        sum_4 += temp2_g[(xp+k)*W + yp+4];
                        sum_5 += temp2_g[(xp+k)*W + yp+5];
                        sum_6 += temp2_g[(xp+k)*W + yp+6];
                        sum_7 += temp2_g[(xp+k)*W + yp+7];
                        sum_8 += temp2_g[(xp+k)*W + yp+8];
                        sum_9 += temp2_g[(xp+k)*W + yp+9];
                        sum_10 += temp2_g[(xp+k)*W + yp+10];
                        sum_11 += temp2_g[(xp+k)*W + yp+11];
                        sum_12 += temp2_g[(xp+k)*W + yp+12];
                        sum_13 += temp2_g[(xp+k)*W + yp+13];
                        sum_14 += temp2_g[(xp+k)*W + yp+14];
                        sum_15 += temp2_g[(xp+k)*W + yp+15];
                        */
                    }
                    scalar weight_0, weight_1, weight_2, weight_3;

                    scalar fweights_0_num = features_weights_r_num[xp * W + yp];
                    scalar fweights_0_den = features_weights_r_den[xp * W + yp];
                    scalar fweights_1_num = features_weights_r_num[xp * W + yp + 1];
                    scalar fweights_1_den = features_weights_r_den[xp * W + yp + 1];
                    scalar fweights_2_num = features_weights_r_num[xp * W + yp + 2];
                    scalar fweights_2_den = features_weights_r_den[xp * W + yp + 2];
                    scalar fweights_3_num = features_weights_r_num[xp * W + yp + 3];
                    scalar fweights_3_den = features_weights_r_den[xp * W + yp + 3];

                    if(sum_0 * fweights_0_den <= neigh_g * fweights_0_num)
                        weight_0 = exp(-fweights_0_num / fweights_0_den);
                    else
                        weight_0 = exp(-sum_0 / neigh_g);
                    if(sum_1 * fweights_1_den <= neigh_g * fweights_1_num)
                        weight_1 = exp(-fweights_1_num / fweights_1_den);
                    else
                        weight_1 = exp(-sum_1 / neigh_g);
                    if(sum_2 * fweights_2_den <= neigh_g * fweights_2_num)
                        weight_2 = exp(-fweights_2_num / fweights_2_den);
                    else
                        weight_2 = exp(-sum_2 / neigh_g);
                    if(sum_3 * fweights_3_den <= neigh_g * fweights_3_num)
                        weight_3 = exp(-fweights_3_num / fweights_3_den);
                    else
                        weight_3 = exp(-sum_3 / neigh_g);

                    weight_sum[1 + 3 * (xp * W + yp)] += weight_0;
                    weight_sum[1 + 3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[1 + 3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[1 + 3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[1 + 3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[1 + 3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[1 + 3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[1 + 3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[1 + 3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[1 + 3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[1 + 3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[1 + 3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[1 + 3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[1 + 3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[1 + 3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[1 + 3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_g[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_g[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_g[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_g[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_g[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_g[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_g[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_g[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_g[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_g[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_g[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_g[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_g[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_g[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_g[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_g[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }
}

void candidate_B_SSA(scalar *output_b, scalar *weight_sum, scalar *color, scalar *features_weights_b, const int r_x, const int r_y, const int R, const int H, const int W) {
            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight_0 = features_weights_b[xp * W + yp];
                    scalar weight_1 = features_weights_b[xp * W + yp+1];
                    scalar weight_2 = features_weights_b[xp * W + yp+2];
                    scalar weight_3 = features_weights_b[xp * W + yp+3];
                    /*
                    scalar weight_4 = features_weights_b[xp * W + yp+4];
                    scalar weight_5 = features_weights_b[xp * W + yp+5];
                    scalar weight_6 = features_weights_b[xp * W + yp+6];
                    scalar weight_7 = features_weights_b[xp * W + yp+7];
                    scalar weight_8 = features_weights_b[xp * W + yp+8];
                    scalar weight_9 = features_weights_b[xp * W + yp+9];
                    scalar weight_10 = features_weights_b[xp * W + yp+10];
                    scalar weight_11 = features_weights_b[xp * W + yp+11];
                    scalar weight_12 = features_weights_b[xp * W + yp+12];
                    scalar weight_13 = features_weights_b[xp * W + yp+13];
                    scalar weight_14 = features_weights_b[xp * W + yp+14];
                    scalar weight_15 = features_weights_b[xp * W + yp+15];
                    */
                    
                    weight_sum[2 + 3 * (xp * W + yp)] += weight_0;
                    weight_sum[2 + 3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[2 + 3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[2 + 3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[2 + 3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[2 + 3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[2 + 3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[2 + 3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[2 + 3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[2 + 3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[2 + 3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[2 + 3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[2 + 3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[2 + 3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[2 + 3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[2 + 3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_b[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_b[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_b[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_b[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_b[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_b[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_b[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_b[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_b[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_b[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_b[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_b[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_b[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_b[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_b[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_b[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }
}

void normalize_R_SSA(scalar *output_r, scalar *weight_sum, const int R, const int W, const int H) {
    for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
        for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {
        
            scalar w_0 = weight_sum[3 * (xp * W + yp)];
            scalar w_1 = weight_sum[3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[3 * (xp * W + yp+3)];
            /*
            scalar w_4 = weight_sum[3 * (xp * W + yp+4)];
            scalar w_5 = weight_sum[3 * (xp * W + yp+5)];
            scalar w_6 = weight_sum[3 * (xp * W + yp+6)];
            scalar w_7 = weight_sum[3 * (xp * W + yp+7)];
            scalar w_8 = weight_sum[3 * (xp * W + yp+8)];
            scalar w_9 = weight_sum[3 * (xp * W + yp+9)];
            scalar w_10 = weight_sum[3 * (xp * W + yp+10)];
            scalar w_11 = weight_sum[3 * (xp * W + yp+11)];
            scalar w_12 = weight_sum[3 * (xp * W + yp+12)];
            scalar w_13 = weight_sum[3 * (xp * W + yp+13)];
            scalar w_14 = weight_sum[3 * (xp * W + yp+14)];
            scalar w_15 = weight_sum[3 * (xp * W + yp+15)];
            */

            for (int i=0; i<3; i++)
                output_r[3 * (xp * W + yp) + i] /= w_0;
            for (int i=0; i<3; i++)
                output_r[3 * (xp * W + yp+1) + i] /= w_1;
            for (int i=0; i<3; i++)
                output_r[3 * (xp * W + yp+2) + i] /= w_2;
            for (int i=0; i<3; i++)
                output_r[3 * (xp * W + yp+3) + i] /= w_3;
        }
    }
}

void normalize_G_SSA(scalar *output_g, scalar *weight_sum, const int R, const int W, const int H) {
    for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
        for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {
        
            scalar w_0 = weight_sum[1 + 3 * (xp * W + yp)];
            scalar w_1 = weight_sum[1 + 3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[1 + 3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[1 + 3 * (xp * W + yp+3)];


            for (int i=0; i<3; i++)
                output_g[3 * (xp * W + yp) + i] /= w_0;
            for (int i=0; i<3; i++)
                output_g[3 * (xp * W + yp+1) + i] /= w_1;
            for (int i=0; i<3; i++)
                output_g[3 * (xp * W + yp+2) + i] /= w_2;
            for (int i=0; i<3; i++)
                output_g[3 * (xp * W + yp+3) + i] /= w_3;

        }
    }
}

void normalize_B_SSA(scalar *output_b, scalar *weight_sum, const int R, const int W, const int H) {
    for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
        for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
        
            scalar w_0 = weight_sum[2 + 3 * (xp * W + yp)];
            scalar w_1 = weight_sum[2 + 3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[2 + 3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[2 + 3 * (xp * W + yp+3)];

            for (int i=0; i<3; i++)
                output_b[3 * (xp * W + yp) + i] /= w_0;
            for (int i=0; i<3; i++)
                output_b[3 * (xp * W + yp+1) + i] /= w_1;
            for (int i=0; i<3; i++)
                output_b[3 * (xp * W + yp+2) + i] /= w_2;
            for (int i=0; i<3; i++)
                output_b[3 * (xp * W + yp+3) + i] /= w_3;
        }
    }
}

void border_cases_SSA(scalar *output_r, scalar *output_g, scalar *output_b, scalar *color, const int W, const int H, const int R) {
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < R + F_R; yp++){
            for (int i = 0; i < 3; i++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + F_R; xp++){
        for (int yp = R + F_R ; yp < H - R - F_R; yp++){
            for (int i = 0; i < 3; i++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < R + F_G; yp++){
            for (int i = 0; i < 3; i++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + F_G; xp++){
        for (int yp = R + F_G ; yp < H - R - F_G; yp++){
            for (int i = 0; i < 3; i++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }
}


void candidate_filtering_all_SSA(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, int R, int W, int H){
    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_buffer_zero(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    allocate_channel(&temp, W, H); 
    allocate_channel(&temp2_r, W, H); 
    allocate_channel(&temp2_g, W, H); 

    // Allocate feature weights buffer
    scalar* features_weights_r_num;
    scalar* features_weights_r_den;
    scalar* features_weights_b;
    allocate_channel(&features_weights_r_num, W, H);
    allocate_channel(&features_weights_r_den, W, H);
    allocate_channel(&features_weights_b, W, H);


    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));
    compute_gradients_SSA(gradients, features, W, H, R);
    
    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*F_R+1)*(2*F_R+1);
    scalar neigh_g = 3*(2*F_G+1)*(2*F_G+1);
    scalar neigh_b = 3*(2*F_B+1)*(2*F_B+1);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
           color_weights_SSA(temp, color, color_var, r_x, r_y, W, H, R);


            // Precompute feature weights
            precompute_features_SSA(features_weights_r_num, features_weights_r_den, features_weights_b, features, features_var, gradients, r_x, r_y, R, W, H);


            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            candidate_R_SSA(output_r, weight_sum, temp, temp2_r, features_weights_r_num, features_weights_r_den, color, r_x, r_y, neigh_r, R, W, H);

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            candidate_G_SSA(output_g, weight_sum, temp, temp2_g, features_weights_r_num, features_weights_r_den, color, r_x, r_y, neigh_g, R, W, H);
            

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------
            candidate_B_SSA(output_b, weight_sum, color, features_weights_b, r_x, r_y,  R, H, W);

        }
    }

    // Final Weight Normalization R
    normalize_R_SSA(output_r, weight_sum, R, W, H);

    // Final Weight Normalization G
    normalize_G_SSA(output_g, weight_sum, R, W, H);

    // Final Weight Normalization B
    normalize_B_SSA(output_b, weight_sum, R, W, H);
   

    // Handline Border Cases 
    // ----------------------------------
    border_cases_SSA(output_r, output_g, output_b, color, W, H, R);
    

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2_r);
    free(temp2_g);
    free(features_weights_r_num);
    free(features_weights_r_den);
    free(features_weights_b);
    free(gradients);

}