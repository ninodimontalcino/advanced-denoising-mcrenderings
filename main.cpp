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
#include "validation.hpp"
#include "memory_mgmt.hpp"
#include "implementations/register.hpp"


#define CYCLES_REQUIRED 1e7
#define REP 1
#define MAX_FUNCS 32
// TODO: define number of flops
#define FLOPS (4.)
#define EPS (1e-4)

using namespace std;

//headers
double get_perf_score(denoise_func f);
double perf_test(denoise_func f, string desc, int flops, scalar* c, scalar* c_var, scalar* features, scalar* features_var, int r, int W, int H);

/* Global vars, used to keep track of student functions */
vector<denoise_func> userFuncs;
vector<string> funcNames;
vector<int> funcFlops;
int numFuncs = 0;

char* stringToCharArray(string string_name){
  char * tab2 = new char [string_name.length()+1];
  strcpy (tab2, string_name.c_str());
  return tab2;
}

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
  bool RMSE = true;

  // ------------------------------------
  // (..) PARAMETER PARSING -> from argv
  // ------------------------------------
  int r;
  string img_size;
  if(argc <= 2) {
    std::cout << "Missing arguments.\nUsage: ./main img_size r" << std::endl;
    return 1;
  }

  r = atoi(argv[2]);
  img_size = argv[1];

  // ------------------------------------
  // (..) BUFFER INIT AND VARIABLE DEF.
  // ------------------------------------

  // Ground Truth buffer
  scalar* gt;

  // Color Buffers and Variance
  scalar* c;
  scalar* c_var;

  // Feature Buffers and Variance
  scalar* f_albedo;
  scalar* f_albedo_var;
  scalar* f_depth;
  scalar* f_depth_var;
  scalar* f_normal;
  scalar* f_normal_var;

  // Other parameters
  int W, H;

  // ------------------------------------
  // (..) FILENAME DEFINITION
  // ------------------------------------
  const string path = "../renderings/100spp/" + img_size;
  const string filename_c = path + "/scene_Coateddiffuse.exr";
  const string filename_varc = path + "/scene_Coateddiffuse_variance.exr";
  const string filename_albedo = path + "/scene_Coateddiffuse_albedo.exr";
  const string filename_albedo_variance = path + "/scene_Coateddiffuse_albedo_variance.exr";
  const string filename_depth = path + "/scene_Coateddiffuse_depth.exr";
  const string filename_depth_variance = path + "/scene_Coateddiffuse_depth_variance.exr";
  const string filename_normal = path + "/scene_Coateddiffuse_normal.exr";
  const string filename_normal_variance = path + "/scene_Coateddiffuse_normal_variance.exr";

  // GT Definition
  const string path_GT = "../renderings/5000spp_GT/" + img_size;
  const string filename_GT = path_GT + "/scene_CoateddiffuseGT.exr";
  
  // ------------------------------------
  // (..) EXR LOADING 
  // ------------------------------------

  // (1) Load main image and its variance
  load_exr(stringToCharArray(filename_c), &c, W, H);
  load_exr(stringToCharArray(filename_varc), &c_var, W, H);

  // (2) Load individual features
  load_exr(stringToCharArray(filename_albedo), &f_albedo, W, H);
  load_exr(stringToCharArray(filename_albedo_variance), &f_albedo_var, W, H);
  load_exr(stringToCharArray(filename_depth), &f_depth, W, H);
  load_exr(stringToCharArray(filename_depth_variance), &f_depth_var, W, H);
  load_exr(stringToCharArray(filename_normal), &f_normal, W, H);
  load_exr(stringToCharArray(filename_normal_variance), &f_normal_var, W, H);

  // (3) Load GT 
  load_exr(stringToCharArray(filename_GT), &gt, W, H);

  // Precompute channel sizue
  int WH = W * H;

  // (3) Feature Stacking
  // => Access Pattern: features[i][x][y] where i in (1:= albedo, 2:= depth, 3:= normal)
  scalar* features;
  scalar* features_var;

  // features = (scalar*) malloc(3 * WH * sizeof(scalar));
  // features_var = (scalar*) malloc(3 * WH * sizeof(scalar));
  allocate_buffer_aligned(&features, W, H);
  allocate_buffer_aligned(&features_var, W, H);

  // (a) Features
  copy(f_albedo, f_albedo + WH, features);
  copy(f_depth, f_depth + WH, features + WH);
  copy(f_normal, f_normal + WH, features + 2*WH);

  // (b) Feature Variances
  copy(f_albedo_var, f_albedo_var + WH, features_var);
  copy(f_depth_var, f_depth_var + WH, features_var + WH);
  copy(f_normal_var, f_normal_var + WH, features_var + 2*WH);

  // DEBUGGING: Output loaded buffer
  if(debug_EXR_loading){
    for (int i = 0; i < H; i ++) {
      for (int j = 0; j < W; j ++) {
        cout << features[0 * WH + j * W + i] << " " << features[1 * WH + j * W + i] << " " << features[2 * WH + j * W + i] << " ";
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
   

  cout << "---------------------------------------------" << endl;
  cout << " (1) Compute Reference Solution" << endl;
  cout << "---------------------------------------------" << endl;

  // Call correct function and check output
  scalar* out_img;
  allocate_buffer_aligned(&out_img, W, H);
  //out_img = (scalar*) malloc(3 * WH * sizeof(scalar));

  denoise_func f = userFuncs[0];
  f(out_img, c, c_var, features, features_var, r, W, H);

  // Compute RMSE 
  double _rmse = rmse(out_img, gt, W, H);
  if (RMSE){
    printf("RMSE: %f \n\n", _rmse);
  }


  // Run functions and check if they produce the same output as the Vanilla Implementation 

  cout << "---------------------------------------------" << endl;
  cout << " (2) Validating optimized functions" << endl;
  cout << "---------------------------------------------" << endl;

  scalar* out_img_f;
  allocate_buffer_aligned(&out_img_f, W, H);
  //out_img_f = (scalar*) malloc(3 * WH * sizeof(scalar));

  // Only run for optimized functions => don't repeat vanilla computation
  for (i = 1; i < numFuncs; i++) {
    cout << endl << "Validating: " << funcNames[i] << endl;
    denoise_func f = userFuncs[i];
    f(out_img_f, c, c_var, features, features_var, r, W, H);

    double _rmse2 = rmse(out_img_f, gt, W, H);
    if (RMSE){
      printf("RMSE: %f \n", _rmse2);
    }

    //Compare out_img_f with out_img_f
    //if (!compare_buffers(out_img, out_img_f, W, H)){
    double error[4];
    maxAbsError(error, out_img, out_img_f, W, H);
    if (abs(_rmse -_rmse2) > EPS){
      printf("Function %d produces a different result! \n", i);
      printf("Abs. Max. Buffer Difference: %f at position: [%f][%f][%f] \n", error[0], error[1], error[2], error[3]);
    }

  }

  free(out_img);
  free(out_img_f);

  // Performance Testing
  cout << "---------------------------------------------" << endl;
  cout << " (3) Performance Tests (including warmup)" << endl;
  cout << "---------------------------------------------" << endl;
  for (i = 0; i < numFuncs; i++)
  {
    perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i], c, c_var, features, features_var, r, W, H);
    cout << endl << "Running: " << funcNames[i] << endl;
    cout << perf << " cycles" << endl;
  }
  
  return 0;
}


/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(denoise_func f, string name, int flops)
{
  userFuncs.push_back(f);
  funcNames.emplace_back(name);
  funcFlops.push_back(flops);

  numFuncs++;
}

/*
* Checks the given function for validity. If valid, then computes and
* reports and returns the number of cycles required per iteration
*/
double perf_test(denoise_func f, string desc, int flops, scalar* c, scalar* svar_c, scalar* features, scalar* svar_f, int r, int W, int H)
{
  int WH = W*H;

  double cycles = 0.;
  double perf = 0.0;
  long num_runs = 1;
  double multiplier = 1;
  myInt64 start, end;

  // Init Buffer for output
  scalar* out_img_perf;
  out_img_perf = (scalar*) malloc(3 * WH * sizeof(scalar));

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(out_img_perf, c, svar_c, features, svar_f, r, W, H);      
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
      f(out_img_perf, c, svar_c, features, svar_f, r, W, H);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;

    cyclesList.push_back(cycles);
    perfList.push_back(FLOPS / cycles);
  }

  free(out_img_perf);

  cyclesList.sort();
  cycles = cyclesList.front();  
  return  cycles;
}
