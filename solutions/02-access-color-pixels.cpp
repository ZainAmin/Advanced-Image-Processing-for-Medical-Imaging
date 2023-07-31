// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{
		// create a 1000x1000 sized matrix, 3 channel (color), 8-bits, initialized to yellow
		cv::Mat aMat(500, 500, CV_8UC(3), cv::Scalar(0,255,255));
		printf("Image created: dims = %d x %d, channels = %d, bitdepth = %d\n",
			aMat.rows, aMat.cols, aMat.channels(), aia::bitdepth(aMat.depth()));
		aia::imshow("A 8-bit color image", aMat, true);
		//                                  /\  
		//                                  ||  
		//                                  || wait for the user to press a key before closing the window

		// exercise: we want to draw the Italian flag
		// goal:     learn how to access color pixels


		// first method: use strict raw pointer approach (hardest/fastest)
		// (not explained in the class: skip this if you find it too difficult)
		for(int y=0; y<aMat.rows; y++)
		{
			unsigned char *ythRow =  aMat.ptr<unsigned char>(y);

			for(int x=0; x<aMat.cols; x++)
			{
				// so far, same as for grayscale images...
				// however, in a color image each pixel is a RGB triple (R,G,B)
				//
				// OpenCV (as many other libraries) adopts an 'interleaved' representation of multi-channel images: 
				// - any image is represented by a of a rows x cols matrix of pixels
				// - pixels are made by as many values as the number of channels
				// - in a grayscale image,  1 pixel <-> 1 value
				// - in a n-channel image,  1 pixel <-> n values (i.e., a vector !!!)
				// For instance, a 'noninterleaved' representation of a n-channel image would consist of a
				// 3D matrix of size n_channels x n_rows x n_cols

				// ...in addition
				// OpenCV (by default) loads color image in BGR color space instead of RGB color space
				// (Blue, Green, Red) triplets instead of (Red, Green, Blue)

				// Thus, instead of ythRow[x] that we used to access pixel (y,x) in grayscale images, we now have
				// ythRow[3*x + 0]  to access the B(lue)  channel at pixel (y,x)
				// ythRow[3*x + 1]  to access the G(reen) channel at pixel (y,x)
				// ythRow[3*x + 2]  to access the R(ed)   channel at pixel (y,x)
				// ...yes, but why?
				// ...we have to look at how pixels are stored in memory,
				// in a grayscale image, for each row y:
				//  x=0                           x=1                            ...  x=cols-1
				// |          I(0,y)             |           I(1,y)            | ... |             I(cols-1,y) 
				//              ||                             ||                                      ||
				//          ythRow[0]                       ythRow[1]                            ythRow[cols-1]   
				//
				// in a color image, for each row y:
				//  x=0                           x=1                            ...  x=cols-1       
				// | Ib(0,y)   Ig(0,y)   Ir(0,y) | Ib(1,y)   Ig(1,y)   Ir(1,y) | ... |    Ib(cols-1,y)       Ig(cols-1,y)           Ir(cols-1,y) 
				//      ||       ||        ||        ||        ||        ||                    ||                 ||                    ||
				//  ythRow[0] ythRow[1] ythRow[2] ythRow[3] ythRow[4] ythRow[5]  ...   ythRow[3*(cols-1)]  ythRow[3*(cols-1)+1]  ythRow[3*(cols-1)+2]
				// 
				// where Ib = intensity of blue channel, Ig = intensity of green channel, Ir = intensity of red channel

				// in addition, here we want to draw the Italian flag, which has three equally-spaced vertical bands:
				// Green, White, and Red
				// GREEN band = (0,255,0) in the BGR color space
				if(x < aMat.cols / 3.0f)
				{
					ythRow[3*x + 0] = 0;	// B channel
					ythRow[3*x + 1] = 255;	// G channel
					ythRow[3*x + 2] = 0;	// R channel
				}
				// WHITE band = (255,255,255) in the BGR color space
				else if(x > aMat.cols / 3.0f && x < 2.0f * aMat.cols / 3)
				{
					ythRow[3*x + 0] = 255;	// B channel
					ythRow[3*x + 1] = 255;	// G channel
					ythRow[3*x + 2] = 255;	// R channel
				}
				// RED band = (0,0,255) in the BGR color space
				else
				{
					ythRow[3*x + 0] = 0;	// B channel
					ythRow[3*x + 1] = 0;	// G channel
					ythRow[3*x + 2] = 255;	// R channel
				}
			}
		}
		aia::imshow("Italian flag (first method)", aMat);


		// second method: use cv::Vec3b (hard/fast)
		for(int y=0; y<aMat.rows; y++)
		{
			cv::Vec3b* yThRow = aMat.ptr<cv::Vec3b>(y);
			for(int x=0; x<aMat.cols; x++)
			{
				if(x < aMat.cols/3)
				{
					yThRow[x][0] = 0;
					yThRow[x][1] = 255;
					yThRow[x][2] = 0;
				}
				else if(x > 2*aMat.cols/3)
				{
					yThRow[x][0] = 0;
					yThRow[x][1] = 0;
					yThRow[x][2] = 255;
				}
				else
				{
					yThRow[x][0] = 255;
					yThRow[x][1] = 255;
					yThRow[x][2] = 255;
				}
			}
		}
		aia::imshow("Italian flag (second method)", aMat);


		// third method: using ROIs
		aMat.colRange(0, aMat.cols/3).setTo(cv::Scalar(0,255,0));
		aMat.colRange(2*aMat.cols/3, aMat.cols).setTo(cv::Scalar(0,0,255));
		aia::imshow("Italian flag (third method)", aMat);


		// fourth method: using split/merge channels and ROIs
		// In some cases, it could be more useful to access/process the cv::Mat in its 'noninterleaved' form,
		// i.e. in which channels are explicitly separated and considered as the 3rd dimension
		std::vector<cv::Mat> imgChannels;	// a vector of 2D matrices ( = 3D matrix )
		cv::split(aMat, imgChannels);	// splits a multichannel image into a vector of single-channels images

		// blue channel
		imgChannels[0].setTo(cv::Scalar(0));
		//               /\        /\
		//               ||        || 0 intensity (black)
		//               ||  set all pixels to a certain value     
		imgChannels[0].colRange(0, imgChannels[0].cols / 3.0f).setTo(0);
		//               /\
		//               ||  column range selection (there also is the rowRange() for row range selection)
		imgChannels[0].colRange(imgChannels[0].cols / 3.0f, 2.0f*imgChannels[0].cols / 3.0f).setTo(255);
		imgChannels[0].colRange(2*imgChannels[0].cols / 3.0f, imgChannels[0].cols-1).setTo(0);

		// green channel
		imgChannels[1].setTo(cv::Scalar(0));
		imgChannels[1].colRange(0, imgChannels[1].cols / 3.0f).setTo(255);
		imgChannels[1].colRange(imgChannels[1].cols / 3.0f, 2.0f*imgChannels[1].cols / 3.0f).setTo(255);
		imgChannels[1].colRange(2*imgChannels[1].cols / 3.0f, imgChannels[1].cols-1).setTo(0);

		// red channel
		imgChannels[2].setTo(cv::Scalar(0));
		imgChannels[2].colRange(0, imgChannels[2].cols / 3.0f).setTo(0);
		imgChannels[2].colRange(imgChannels[2].cols / 3.0f, 2.0f*imgChannels[2].cols / 3.0f).setTo(255);
		imgChannels[2].colRange(2*imgChannels[2].cols / 3.0f, imgChannels[2].cols-1).setTo(255);

		cv::merge(imgChannels, aMat);
		aia::imshow("Italian flag (fourth method)", aMat);

		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

