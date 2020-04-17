#pragma once

#include <ImfRgbaFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfArray.h>
#include "flt.hpp"

//#include "drawImage.h"
//#include "namespaceAlias.h"

#include <iostream>
#include <algorithm>

#include <ImfNamespace.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;

using namespace IMF;
using namespace std;
using namespace IMATH_NAMESPACE;


/*! ------------------------------------
	EXR Loading 
	-> Loads and return an EXR image
	-> Access-Pattern buf[{r,g,b}][x][y]
    
	Parameters:
		- filename 		EXR-Filename
    	- buf 	    	Pointer to destination buffer
		- img_width 	Image Width
		- img_height	Image Height

	Returns
		- buf			buffer containing loaded EXR image

*/
void load_exr(const char fileName[], buffer* buf, int &img_width, int &img_eight);				

/*! ------------------------------------
	EXR Writing
	-> writes an EXR image
	Parameters:
		- filename 		EXR-Filename
    	- buf 	    	Buffer (containing image to read to .exr)
		- img_width 	Image Width
		- img_height	Image Height

	Returns
		--> write an EXR file

*/
void write_buffer_exr(const char fileName[],  buffer* buf, int img_width, int img_height);
void write_channel_exr(const char fileName[], channel* buf, int img_width, int img_height);


// ------------------------------------------------------------------------------------------------------------------
// Some Helper Function for EXR handling

void readGZ1 (const char fileName[], Array2D<float> &rPixels, Array2D<float> &gPixels, Array2D<float> &zPixels, int &width, int &height);