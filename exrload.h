#pragma once

#include <ImfRgbaFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>

#include <iostream>
#include <algorithm>

#include <ImfNamespace.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;

using namespace IMF;
using namespace std;
using namespace IMATH_NAMESPACE;

void
readRgba1 (const char fileName[],
	   Array2D<Rgba> &pixels,
	   int &width,
	   int &height);