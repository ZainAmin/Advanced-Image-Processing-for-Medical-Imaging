// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace aia
{
	// local (8-neighborhood) region-growing segmentation based on color difference
	void colorDiffSegmentation(const cv::Mat & img, cv::Scalar colorDiff = cv::Scalar::all(3));
}

// GOAL: Pectoral Muscle Segmentation with Mean-Shift
int main() 
{
	try
	{
		// load image
		// pro_mammogram_1, pro_mammogram_2, or pro_mammogram_3
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pro_mammogram_3.tif");
		if(!img.data)
			throw aia::error("Cannot open image");


		// downscale image (too big = mean-shift too computationally heavy)
		cv::resize(img, img, cv::Size(0,0), 0.2, 0.2, cv::INTER_AREA);
		aia::imshow("Image", img);

		// trace a line to avoid region-growing breaking the bottom part of the pectoral
		// muscle and filling the rest of the breast
		// cv::line(img, cv::Point(0, img.rows/2-100),  cv::Point(img.cols-1, img.rows/2-100), cv::Scalar(255));

		// mean shift filtering
		cv::Mat img_ms;
		cv::pyrMeanShiftFiltering(img, img_ms, 20, 20, 0);
		aia::imshow("Mean-Shift", img_ms);

		// region growing based on color difference
		aia::colorDiffSegmentation(img_ms);
		aia::imshow("Postprocessing", img_ms);

		// convert to grayscale
		cv::cvtColor(img_ms, img_ms, cv::COLOR_BGR2GRAY);
		aia::imshow("Grayscale", img_ms);

		// median blur
		cv::medianBlur(img_ms, img_ms, 7);
		aia::imshow("Median", img_ms);

		// select top-right connected component
		unsigned char muscle_intensity = img_ms.at<unsigned char>(10, img_ms.cols-10);
		cv::inRange(img_ms, muscle_intensity, muscle_intensity, img_ms);
		std::vector < std::vector <cv::Point> > components;
		cv::findContours(img_ms, components, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		int muscle_component_idx = 0;
		if(components.size() != 1)
		{
			double maxArea = 0;
			for(int i=0; i<components.size(); i++)
			{
				double compArea = cv::contourArea(components[i]);
				if(compArea > maxArea)
				{
					maxArea = compArea;
					muscle_component_idx = i;
				}
			}
		}

		// overlay with original image
		cv::Mat selection_layer = img.clone();
		cv::drawContours(selection_layer, components, muscle_component_idx, cv::Scalar(0, 255, 255), CV_FILLED, CV_AA);
		cv::addWeighted(img, 0.8, selection_layer, 0.2, 0, img);
		aia::imshow("Result", img);

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

// local (8-neighborhood) region-growing segmentation based on color difference
void aia::colorDiffSegmentation(const cv::Mat & img, cv::Scalar colorDiff)
{
	// a number generator we will use to assign random colors to
	cv::RNG rng = cv::theRNG();

	// mask used to accelerate processing
	cv::Mat mask( img.rows+2, img.cols+2, CV_8UC1, cv::Scalar::all(0) );

	// use every pixel as seed for region growing
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			// avoid growing from a seed that has already been merged with another region
			if( mask.at<uchar>(y+1, x+1) == 0 )
			{
				cv::Scalar newVal( rng(256), rng(256), rng(256) );
				cv::floodFill( img, mask, cv::Point(x,y), newVal, 0, colorDiff, colorDiff );
			}
		}
	}
}