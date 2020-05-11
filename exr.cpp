#include "exr.h"
#include "memory_mgmt.hpp"


void load_exr(const char fileName[], scalar** buf, int &W, int &H) 
{
        
        // Read RGB Channels
        Array2D<float> pixelsR, pixelsG, pixelsB;
        readGZ1(fileName, pixelsR, pixelsG, pixelsB, W, H);

        int WH = W * H;

        // Memory Allocation
        (*buf) = (scalar*) malloc(3 * W * H * sizeof(scalar));

        // Fill buffer with corresponding values -> c[channel][x][y]
        for (int i = 0; i < W; i++) {
            for (int j=0; j<H; j++) {
                (*buf)[3 * (i * W + j) + 0] = pixelsR[j][i];
                (*buf)[3 * (i * W + j) + 1] = pixelsG[j][i];
                (*buf)[3 * (i * W + j) + 2] = pixelsB[j][i];
            }
        }
}


void write_buffer_exr(const char fileName[], scalar* buf, int W, int H){

    int WH = W * H;

    // Write information from buffer in 2DArrays
    Array2D<float> pixelsR, pixelsG, pixelsB;
    pixelsR.resizeErase(H, W);
    pixelsG.resizeErase(H, W);
    pixelsB.resizeErase(H, W);

    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){
            pixelsR[y][x] = buf[3 * (x * W + y) + 0];
            pixelsG[y][x] = buf[3 * (x * W + y) + 1];
            pixelsB[y][x] = buf[3 * (x * W + y) + 2];
        }
    }


    // Defining Header => we write 3 channels (R,G,B)
    Header header (W, H); 
    header.channels().insert ("R", Channel (IMF::FLOAT)); 
    header.channels().insert ("G", Channel (IMF::FLOAT));
    header.channels().insert ("B", Channel (IMF::FLOAT)); 

    // Init output file
    OutputFile file (fileName, header);

    // Create FrameBuffer
    FrameBuffer frameBuffer;

    frameBuffer.insert ("R",
                        Slice (IMF::FLOAT,
                        (char *) &pixelsR[0][0],                        
                        sizeof (pixelsR[0][0]) * 1,      
                        sizeof (pixelsR[0][0]) * W            
                        ));

    frameBuffer.insert ("G",
                        Slice (IMF::FLOAT,
                        (char *) &pixelsG[0][0],                        
                        sizeof (pixelsG[0][0]) * 1,      
                        sizeof (pixelsG[0][0]) * W  
                        ));

    frameBuffer.insert ("B",
                        Slice (IMF::FLOAT,
                        (char *) &pixelsB[0][0],                        
                        sizeof (pixelsB[0][0]) * 1,      
                        sizeof (pixelsB[0][0]) * W    
                        ));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(H);         

}

void write_channel_exr(const char fileName[], scalar* c, int W, int H){

    // Write information from buffer in 2DArrays
    Array2D<float> pixels;
    pixels.resizeErase(H, W);

    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){
            pixels[y][x] = c[x * W + y];
        }
    }

    // Defining Header => we write 3 channels (R,G,B)
    Header header (W, H); 
    header.channels().insert ("R", Channel (IMF::FLOAT)); 
    header.channels().insert ("G", Channel (IMF::FLOAT));
    header.channels().insert ("B", Channel (IMF::FLOAT)); 

    // Init output file
    OutputFile file (fileName, header);

    // Create FrameBuffer
    FrameBuffer frameBuffer;

    frameBuffer.insert ("R",
                        Slice (IMF::FLOAT,
                        (char *) &pixels[0][0],                        
                        sizeof (pixels[0][0]) * 1,      
                        sizeof (pixels[0][0]) * W            
                        ));

    frameBuffer.insert ("G",
                        Slice (IMF::FLOAT,
                        (char *) &pixels[0][0],                        
                        sizeof (pixels[0][0]) * 1,      
                        sizeof (pixels[0][0]) * W  
                        ));

    frameBuffer.insert ("B",
                        Slice (IMF::FLOAT,
                        (char *) &pixels[0][0],                        
                        sizeof (pixels[0][0]) * 1,      
                        sizeof (pixels[0][0]) * W    
                        ));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(H);         

}



void readGZ1 (const char fileName[], Array2D<float> &rPixels, Array2D<float> &gPixels, Array2D<float> &zPixels, int &width, int &height)
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
