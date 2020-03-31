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

void
load_image (const char fileName[],
	   float*** &pixels,
	   int &width,
	   int &height);

void
readRgba1 (const char fileName[],
	   Array2D<Rgba> &pixels,
	   int &width,
	   int &height);

void
readGZ1 (const char fileName[],
	 Array2D<float> &rPixels,
	 Array2D<float> &gPixels,
	 Array2D<float> &zPixels,
	 int &width, int &height);