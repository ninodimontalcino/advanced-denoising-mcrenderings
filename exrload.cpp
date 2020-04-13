#include "exrload.h"

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
                buffer* buf,
	            int &img_width, 
                int &img_height) 
{
        
        // Read RGB Channels
        Array2D<float> pixelsR, pixelsG, pixelsB;
        readGZ1(fileName, pixelsR, pixelsG, pixelsB, img_width, img_height);

        // Init buffer channels
        (*buf)[0] = (channel)malloc(img_width*sizeof(void*));  
        (*buf)[1] = (channel)malloc(img_width*sizeof(void*));
        (*buf)[2] = (channel)malloc(img_width*sizeof(void*));

        // Fill buffer with corresponding values -> c[channel][x][y]
        for (int i = 0; i < img_width; i ++) {

            (*buf)[0][i] = (scalar*)malloc(img_height*sizeof(scalar));
            (*buf)[1][i] = (scalar*)malloc(img_height*sizeof(scalar));
            (*buf)[2][i] = (scalar*)malloc(img_height*sizeof(scalar));

            for (int j=0; j<img_height; j ++) {

                (*buf)[0][i][j] = pixelsR[j][i];
                (*buf)[1][i][j] = pixelsG[j][i];
                (*buf)[2][i][j] = pixelsB[j][i];
            }
        }
}

