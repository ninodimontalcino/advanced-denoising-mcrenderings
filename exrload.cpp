#include "exrload.h"

void
readRgba1 (const char fileName[],
	   Array2D<Rgba> &pixels,
	   int &width,
	   int &height)
{
    //
    // Read an RGBA image using class RgbaInputFile:
    //
    //	- open the file
    //	- allocate memory for the pixels
    //	- describe the memory layout of the pixels
    //	- read the pixels from the file
    //

    RgbaInputFile file (fileName);
    Box2i dw = file.dataWindow();

    width  = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    pixels.resizeErase (height, width);

    file.setFrameBuffer (&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels (dw.min.y, dw.max.y);
}

// void
// readRGB (const char fileName[],
// 	 float &rPixels,
// 	 float &gPixels,
// 	 float &zPixels,
// 	 int &width, int &height)
// {

//     InputFile file (fileName);

//     Box2i dw = file.header().dataWindow();
//     width  = dw.max.x - dw.min.x + 1;
//     height = dw.max.y - dw.min.y + 1;

//     rPixels.resizeErase (height, width);
//     gPixels.resizeErase (height, width);
//     zPixels.resizeErase (height, width);

//     FrameBuffer frameBuffer;

//     frameBuffer.insert ("R",					// name
// 			Slice (IMF::FLOAT,			// type
// 			       (char *) (&rPixels[0][0] -	// base
// 					 dw.min.x -
// 					 dw.min.y * width),
// 			       sizeof (rPixels[0][0]) * 1,	// xStride
// 			       sizeof (rPixels[0][0]) * width,	// yStride
// 			       1, 1,				// x/y sampling
// 			       FLT_MAX));				// fillValue

//     frameBuffer.insert ("G",					// name
// 			Slice (IMF::FLOAT,			// type
// 			       (char *) (&gPixels[0][0] -	// base
// 					 dw.min.x -
// 					 dw.min.y * width),
// 			       sizeof (gPixels[0][0]) * 1,	// xStride
// 			       sizeof (gPixels[0][0]) * width,	// yStride
// 			       1, 1,				// x/y sampling
// 			       FLT_MAX));				// fillValue

//     frameBuffer.insert ("Z",					// name
// 			Slice (IMF::FLOAT,			// type
// 			       (char *) (&zPixels[0][0] -	// base
// 					 dw.min.x -
// 					 dw.min.y * width),
// 			       sizeof (zPixels[0][0]) * 1,	 // xStride
// 			       sizeof (zPixels[0][0]) * width,	// yStride
// 			       1, 1,				// x/y sampling
// 			       FLT_MAX));			// fillValue

//     file.setFrameBuffer (frameBuffer);
//     file.readPixels (dw.min.y, dw.max.y);
// }

void
readGZ1 (const char fileName[],
	 Array2D<float> &rPixels,
	 Array2D<float> &gPixels,
	 Array2D<float> &zPixels,
	 int &width, int &height)
{

    InputFile file (fileName);

    Box2i dw = file.header().dataWindow();
    width  = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;

    rPixels.resizeErase (height, width);
    gPixels.resizeErase (height, width);
    zPixels.resizeErase (height, width);

    FrameBuffer frameBuffer;

    frameBuffer.insert ("R",					// name
			Slice (IMF::FLOAT,			// type
			       (char *) (&rPixels[0][0] -	// base
					 dw.min.x -
					 dw.min.y * width),
			       sizeof (rPixels[0][0]) * 1,	// xStride
			       sizeof (rPixels[0][0]) * width,	// yStride
			       1, 1,				// x/y sampling
			       FLT_MAX));				// fillValue

    frameBuffer.insert ("G",					// name
			Slice (IMF::FLOAT,			// type
			       (char *) (&gPixels[0][0] -	// base
					 dw.min.x -
					 dw.min.y * width),
			       sizeof (gPixels[0][0]) * 1,	// xStride
			       sizeof (gPixels[0][0]) * width,	// yStride
			       1, 1,				// x/y sampling
			       FLT_MAX));				// fillValue

    frameBuffer.insert ("B",					// name
			Slice (IMF::FLOAT,			// type
			       (char *) (&zPixels[0][0] -	// base
					 dw.min.x -
					 dw.min.y * width),
			       sizeof (zPixels[0][0]) * 1,	 // xStride
			       sizeof (zPixels[0][0]) * width,	// yStride
			       1, 1,				// x/y sampling
			       FLT_MAX));			// fillValue

    file.setFrameBuffer (frameBuffer);
    file.readPixels (dw.min.y, dw.max.y);
}

void load_image(const char fileName[],
     buffer** &c,
	 int &width, int &height) 
{
        Array2D<float> pixelsR, pixelsG, pixelsB;
        int w, h;
        readGZ1(fileName, pixelsR, pixelsG, pixelsB, w, h);

        cout << w << " " << h << "\n";

        c = new buffer[3];
        c[0] = new float*[h];
        c[1] = new float*[h];
        c[2] = new float*[h];
        for (int i = 0; i < h; i ++) {
            c[0][i] = pixelsR[i];
            c[1][i] = pixelsG[i];
            c[2][i] = pixelsB[i];
        }
}

