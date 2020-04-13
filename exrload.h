#pragma once

#include <ImfRgbaFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfArray.h>
#include "flt.h"

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
	EXR LOADING
	------------------------------------
    \param filename EXR-Filename
    \param pixels 	Destination Buffer
	\param width 	Image Width
	\param height	Image Height

    \return void --> fills pixels-buffer 
*/
void load_image(const char fileName[],		
	   			buffer* &pixels,			
	   			int &width,					
	   			int &height);				


void readGZ1 (const char fileName[],
	 		  Array2D<float> &rPixels,
			  Array2D<float> &gPixels,
	 		  Array2D<float> &zPixels,
	 		  int &width, 
			  int &height);