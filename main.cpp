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
#include "denoise.h"
#include "tsc_x86.h"
#include "exrload.h"
#include <cfloat>


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
double perf_test(denoise_func f, string desc, int flops);
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

  //register_functions();

  // ---------------------------
  // (..) DEBUG VARIABLES
  // ---------------------------
  bool debug_EXR_loading = false;

  // ---------------------------
  // (..) VARIABLE DEFINITION
  // ---------------------------
  buffer *c, *svar_c, *features, *svar_f;
  c = new buffer[3];
  svar_c = new buffer[3];
  features = new buffer[3];
  svar_f = new buffer[3];
  int r, w, h;

  // ---------------------------
  // (..) FILENAME DEFINITION
  // ---------------------------
  const char filename_c[] = "../renderings/100spp/scene_Coateddiffuse.exr";
  const char filename_varc[] = "../renderings/100spp/scene_Coateddiffuse_variance.exr";
  const char filename_albeido[] = "../renderings/100spp/scene_Coateddiffuse_albedo.exr";
  const char filename_varalbeido[] = "../renderings/100spp/scene_Coateddiffuse_albedo_variance.exr";
  const char filename_depth[] = "../renderings/100spp/scene_Coateddiffuse_depth.exr";
  const char filename_depth_variance[] = "../renderings/100spp/scene_Coateddiffuse_depth_variance.exr";
  const char filename_normal[] = "../renderings/100spp/scene_Coateddiffuse_normal.exr";
  const char filename_normal_variance[] = "../renderings/100spp/scene_Coateddiffuse_normal_variance.exr";
  
  // ---------------------------  
  // (..) LOADING 
  // --------------------------- 
  load_image(filename_c, c, w, h);
  load_image(filename_varc, svar_c, w, h);
  load_image(filename_albeido, features, w, h);
  load_image(filename_varalbeido, svar_f, w, h);

  if(debug_EXR_loading){
    for (int i = 0; i < h; i ++) {
      for (int j = 0; j < w; j ++) {
        cout << c[0][j][i] << " " << c[1][j][i] << " " << c[2][j][i] << " ";
      }
      cout << "\n";
    }
  }
  
  // ---------------------------
  // (..) ...
  // ---------------------------


  if (numFuncs == 0){ 
    cout << endl;
    cout << "No functions registered - nothing for driver to do" << endl;
    cout << "Register functions by calling register_func(f, name)" << endl;
    cout << "in register_funcs()" << endl;

    return 0;
  }
  cout << numFuncs << " functions registered." << endl;
   

  // TODO @Nino
  // Call correct function and check output
  buffer *out_img;
  
  denoise_func f = userFuncs[0];
  f(out_img, c, svar_c, features, svar_f, r);
  
  // Store out_img somewhere

  for (i = 0; i < numFuncs; i++) {
    denoise_func f = userFuncs[i];
    f(out_img, c, svar_c, features, svar_f, r);

    // Compute difference between correct and new out_img, check if they are the same
  }


  for (i = 0; i < numFuncs; i++)
  {
    perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i]);
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
  funcNames.__emplace_back(name);
  funcFlops.push_back(flops);

  numFuncs++;
}

/*
* Checks the given function for validity. If valid, then computes and
* reports and returns the number of cycles required per iteration
*/
double perf_test(denoise_func f, string desc, int flops)
{
  double cycles = 0.;
  double perf = 0.0;
  long num_runs = 16;
  double multiplier = 1;
  myInt64 start, end;

  buffer *c, *svar_c, *features, *svar_f;
  int r;
  buffer *out_img;
  // TODO: build inputs

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(out_img, c, svar_c, features, svar_f, r);      
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
      f(out_img, c, svar_c, features, svar_f, r);
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
