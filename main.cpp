/**
 * main.cpp copied from ASL course
 */

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cfloat>

#include "denoise.h"
#include "tsc_x86.h"
#include "exr.h"
#include "memory_mgmt.hpp"
#include "validation.hpp"


#define CYCLES_REQUIRED 1e7
#define REP 10
#define MAX_FUNCS 32
// TODO: define number of flops
#define FLOPS (4.)
#define EPS (1e-3)

using namespace std;

//headers
double get_perf_score(denoise_func f);
void register_functions();
double perf_test(denoise_func f, string desc, int flops, int img_width, int img_height);
void add_function(denoise_func f, string name, int flop);

/* Global vars, used to keep track of student functions */
vector<denoise_func> userFuncs;
vector<string> funcNames;
vector<int> funcFlops;
int numFuncs = 0;

/* void rands(double * m, size_t row, size_t col)
{
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<double> dist(1.0, 5.0);
  for (size_t i = 0; i < row*col; ++i)  
    m[i] = dist(gen);
}
*/

/*
* Main driver routine - calls register_funcs to get student functions, then
* tests all functions registered, and reports the best performance
*/
int main(int argc, char **argv)
{
  //cout << "Starting program. ";
  double perf;
  int i;

  // Function Registering
  register_functions();

  // ------------------------------------
  // (..) DEBUG VARIABLES
  // ------------------------------------
  bool debug_EXR_loading = false;

  // ------------------------------------
  // (..) PARAMETER PARSING -> from argv
  // ------------------------------------
  // TODO: Read all input parameters

  // ------------------------------------
  // (..) BUFFER INIT AND VARIABLE DEF.
  // ------------------------------------

  // Ground Truth buffer
  buffer gt;

  // Color Buffers and Variance
  buffer c, c_var;

  // Feature Buffers and Variance
  buffer f_albedo, f_albedo_var, f_depth, f_depth_var, f_normal, f_normal_var;

  // Other parameters
  int r, img_width, img_height;
  r = 10; // Fixing r=10 for test purposes

  // ------------------------------------
  // (..) FILENAME DEFINITION
  // ------------------------------------
  // TODO: EXR-Loading based on string input such that we can use path + "<..>.exr"
  const string path = "../renderings/100spp/";
  const char filename_GT[] = "../renderings/5000spp_GT/scene_CoateddiffuseGT.exr";
  const char filename_c[] = "../renderings/100spp/scene_Coateddiffuse.exr";
  const char filename_varc[] = "../renderings/100spp/scene_Coateddiffuse_variance.exr";
  const char filename_albedo[] = "../renderings/100spp/scene_Coateddiffuse_albedo.exr";
  const char filename_albedo_variance[] = "../renderings/100spp/scene_Coateddiffuse_albedo_variance.exr";
  const char filename_depth[] = "../renderings/100spp/scene_Coateddiffuse_depth.exr";
  const char filename_depth_variance[] = "../renderings/100spp/scene_Coateddiffuse_depth_variance.exr";
  const char filename_normal[] = "../renderings/100spp/scene_Coateddiffuse_normal.exr";
  const char filename_normal_variance[] = "../renderings/100spp/scene_Coateddiffuse_normal_variance.exr";
  
  // ------------------------------------
  // (..) EXR LOADING 
  // ------------------------------------

 
  // (1) Load main image and its variance
  load_exr(filename_c, &c, img_width, img_height);
  load_exr(filename_varc, &c_var, img_width, img_height);

  // (2) Load individual features
  load_exr(filename_albedo, &f_albedo, img_width, img_height);
  load_exr(filename_albedo_variance, &f_albedo_var, img_width, img_height);
  load_exr(filename_depth, &f_depth, img_width, img_height);
  load_exr(filename_depth_variance, &f_depth_var, img_width, img_height);
  load_exr(filename_normal, &f_normal, img_width, img_height);
  load_exr(filename_normal_variance, &f_normal_var, img_width, img_height);

  // (3) Load GT
  load_exr(filename_GT, &gt, img_width, img_height);

  // (3) Feature Stacking
  // => Access Pattern: features[i][x][y] where i in (1:= albedo, 2:= depth, 3:= normal)
  // => TODO: Fix input features to one channel => or duplicated channel
  buffer features, features_var;
  allocate_buffer(&features, img_width, img_height);
  allocate_buffer(&features_var, img_width, img_height);

  // (a) Features
  features[0] = f_albedo[0];
  features[1] = f_depth[0];
  features[2] = f_normal[0];

  // (b) Feature Variances
  features_var[0] = f_albedo_var[0];
  features_var[1] = f_depth_var[0];
  features_var[2] = f_normal_var[0];

  // DEBUGGING: Output loaded buffer
  if(debug_EXR_loading){
    for (int i = 0; i < img_height; i ++) {
      for (int j = 0; j < img_width; j ++) {
        cout << features[0][j][i] << " " << features[1][j][i] << " " << features[2][j][i] << " ";
      }
      cout << "\n";
    }
  }
  
  // ------------------------------------
  // (..) ...
  // ------------------------------------


  if (numFuncs == 0){ 
    cout << endl;
    cout << "No functions registered - nothing for driver to do" << endl;
    cout << "Register functions by calling register_func(f, name)" << endl;
    cout << "in register_funcs()" << endl;

    return 0;
  }
  cout << numFuncs << " functions registered." << endl;
   
  // Call correct function and check output
  buffer out_img;
  allocate_buffer(&out_img, img_width, img_height);
  denoise_func f = userFuncs[0];
  f(out_img, c, c_var, features, features_var, r, img_width, img_height);

  // Compute RMSE between denoised image and GT (of Vanilla Implementation)
  scalar _rmse = rmse(out_img, gt, img_width, img_height);
  printf("RMSE: %f \n", _rmse);


  // Run functions and check if they produce the same output as the Vanilla Implementation 
  buffer out_img_f;
  allocate_buffer(&out_img_f, img_width, img_height);

  for (i = 0; i < numFuncs; i++) {
    denoise_func f = userFuncs[i];
    f(out_img_f, c, c_var, features, features_var, r, img_width, img_height);

    // Compare out_img_f with out_img_f
    //if (!compare_buffers(out_img, out_img_f)){
    //  printf("Function %d produces a different result! \n", i);
    //}

  }

  // Performance Testing
  /*
  for (i = 0; i < numFuncs; i++)
  {
    perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i], img_width, img_height);
    cout << endl << "Running: " << funcNames[i] << endl;
    cout << perf << " cycles" << endl;
  }
  */
  return 0;
}


/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(denoise_func f, string name, int flops)
{
  userFuncs.push_back(f);
  funcNames.__emplace_back(name);
  funcFlops.push_back(flops);

  numFuncs++;
}

/*
* Checks the given function for validity. If valid, then computes and
* reports and returns the number of cycles required per iteration
*/
double perf_test(denoise_func f, string desc, int flops, int img_width, int img_height)
{
  double cycles = 0.;
  double perf = 0.0;
  long num_runs = 16;
  double multiplier = 1;
  myInt64 start, end;

  buffer c, svar_c, features, svar_f;
  int r;
  buffer out_img;
  // TODO: build inputs

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(out_img, c, svar_c, features, svar_f, r, img_width, img_height);      
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);
    
  } while (multiplier > 2);

  list< double > cyclesList, perfList;

  // Actual performance measurements repeated REP times.
  // We simply store all results and compute medians during post-processing.
  for (size_t j = 0; j < REP; j++) {

    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(out_img, c, svar_c, features, svar_f, r, img_width, img_height);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;

    cyclesList.push_back(cycles);
    perfList.push_back(FLOPS / cycles);
  }

  cyclesList.sort();
  cycles = cyclesList.front();  
  return  cycles;
}
