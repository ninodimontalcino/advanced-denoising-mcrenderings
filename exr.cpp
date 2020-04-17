#include "exr.h"
#include "memory_mgmt.hpp"


void load_exr(const char fileName[], buffer* buf, int &img_width, int &img_height) 
{
        
        // Read RGB Channels
        Array2D<float> pixelsR, pixelsG, pixelsB;
        readGZ1(fileName, pixelsR, pixelsG, pixelsB, img_width, img_height);

        // Memory Allocation
        allocate_buffer(buf, img_width, img_height);

        // Fill buffer with corresponding values -> c[channel][x][y]
        for (int i = 0; i < img_width; i++) {
            for (int j=0; j<img_height; j++) {
                (*buf)[0][i][j] = pixelsR[j][i];
                (*buf)[1][i][j] = pixelsG[j][i];
                (*buf)[2][i][j] = pixelsB[j][i];
            }
        }
}


void write_buffer_exr(const char fileName[], buffer* buf, int img_width, int img_height){

    // Write information from buffer in 2DArrays
    Array2D<float> pixelsR, pixelsG, pixelsB;
    pixelsR.resizeErase(img_height, img_width);
    pixelsG.resizeErase(img_height, img_width);
    pixelsB.resizeErase(img_height, img_width);

    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){
            pixelsR[y][x] = (*buf)[0][x][y];
            pixelsG[y][x] = (*buf)[1][x][y];
            pixelsB[y][x] = (*buf)[2][x][y];
        }
    }


    // Defining Header => we write 3 channels (R,G,B)
    Header header (img_width, img_height); 
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
                        sizeof (pixelsR[0][0]) * img_width            
                        ));

    frameBuffer.insert ("G",
                        Slice (IMF::FLOAT,
                        (char *) &pixelsG[0][0],                        
                        sizeof (pixelsG[0][0]) * 1,      
                        sizeof (pixelsG[0][0]) * img_width  
                        ));

    frameBuffer.insert ("B",
                        Slice (IMF::FLOAT,
                        (char *) &pixelsB[0][0],                        
                        sizeof (pixelsB[0][0]) * 1,      
                        sizeof (pixelsB[0][0]) * img_width    
                        ));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(img_height);         

}


void write_channel_exr(const char fileName[], channel* c, int img_width, int img_height){

    // Write information from buffer in 2DArrays
    Array2D<float> pixels;
    pixels.resizeErase(img_height, img_width);

    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_width; y++){
            pixels[y][x] = (*c)[x][y];
        }
    }

    // Defining Header => we write 3 channels (R,G,B)
    Header header (img_width, img_height); 
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
                        sizeof (pixels[0][0]) * img_width            
                        ));

    frameBuffer.insert ("G",
                        Slice (IMF::FLOAT,
                        (char *) &pixels[0][0],                        
                        sizeof (pixels[0][0]) * 1,      
                        sizeof (pixels[0][0]) * img_width  
                        ));

    frameBuffer.insert ("B",
                        Slice (IMF::FLOAT,
                        (char *) &pixels[0][0],                        
                        sizeof (pixels[0][0]) * 1,      
                        sizeof (pixels[0][0]) * img_width    
                        ));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(img_height);         

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
