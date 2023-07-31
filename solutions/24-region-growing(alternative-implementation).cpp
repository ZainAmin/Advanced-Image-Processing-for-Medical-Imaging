// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace aia
{
	// utility function that rotates 'img' by step*90°
	// step = 0 --> no rotation
	// step = 1 --> 90° CW rotation
	// step = 2 --> 180° CW rotation
	// step = 3 --> 270° CW rotation
	cv::Mat rotate90(cv::Mat img, int step)
	{
		cv::Mat img_rot;

		// adjust step in case it is negative
		if(step < 0)
			step = -step;
		// adjust step in case it exceeds 4
		step = step%4;

		// no rotation
		if(step == 0)
			img_rot = img;
		// 90° CW rotation
		else if(step == 1)
		{
			cv::transpose(img, img_rot);
			cv::flip(img_rot, img_rot, 1);
		}
		// 180° CW rotation
		else if(step == 2)
			cv::flip(img, img_rot, -1);
		// 270° CW rotation
		else if(step == 3)
		{
			cv::transpose(img, img_rot);
			cv::flip(img_rot, img_rot, 0);
		}

		return img_rot;
	}

	// utility function that calculates the per-pixel
	// standard deviation in a ksize x ksize neighborhood
	cv::Mat imstdev(const cv::Mat & img, int ksize)
	{
		// perform all operations with float-precision
		cv::Mat imgF;
		img.convertTo(imgF, CV_32F);

		// create averaging kernel
		cv::Mat kernel = cv::Mat::ones(ksize, ksize, CV_32F);
		kernel /= ksize*ksize;

		// calculate standard deviation
		cv::Mat img_avg, img_avg_sq, img_sq_avg, img_sq, img_stdev;
		cv::filter2D(imgF, img_avg, CV_32F, kernel);
		cv::pow(img_avg, 2, img_avg_sq);
		cv::pow(imgF, 2, img_sq);
		cv::filter2D(img_sq, img_sq_avg, CV_32F, kernel);
		cv::sqrt(img_sq_avg-img_avg_sq, img_stdev);

		return img_stdev;
	}
}

// GOAL: region growing in lightning image
int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");
		float scaling_factor = 0.5;
		aia::imshow("Image", img, true, scaling_factor);
		//cv::imwrite("C:/work/growing_orig.png", img);

		// set parameters
		int thresholdSeed = 230;
		int thresholdPredImg = 110;
		int thresholdPredImgStd = 30;
		int stdev_kernel_size = 5;

		// get seeds image
		cv::Mat seeds;
		cv::threshold(img, seeds, thresholdSeed, 255, cv::THRESH_BINARY);

		// get predicate images
		cv::Mat imgPred1, imgPred2;
		cv::threshold(img, imgPred1, thresholdPredImg, 255, cv::THRESH_BINARY);
		cv::Mat imstd = aia::imstdev(img, stdev_kernel_size);
		cv::normalize(imstd, imstd, 0, 255, cv::NORM_MINMAX);
		imstd.convertTo(imstd, CV_8U);
		cv::threshold(imstd, imgPred2, thresholdPredImgStd, 255, cv::THRESH_BINARY);

		// create SEs to detect candidate points that can grow
		// (rotated SEs will be generated on the fly)
		std::vector <cv::Mat> grow_SEs;
		grow_SEs.push_back((cv::Mat_<char>(3,3) << 
			0,  1,  0, 
			0, -1,  0,
			0,  0,  0 ));		
		grow_SEs.push_back((cv::Mat_<char>(3,3) << 
			0,  0,  1, 
			0, -1,  0,
			0,  0,  0 ));

		// iterative region growing
		cv::Mat seeds_prev;
		cv::Mat candidates(img.rows, img.cols, CV_8U);
		do
		{	
			// generate candidate image
			for(int i=0; i<grow_SEs.size(); i++)
			{
				// perform all 90° rotations so that detection is ani
				for(int j=0; j<4; j++)
				{
					cv::Mat hitormiss;
					cv::morphologyEx(seeds, hitormiss, cv::MORPH_HITMISS, aia::rotate90(grow_SEs[i], j));
					candidates += hitormiss;
				}
			}

			// make a backup copy of the previous seeds
			seeds_prev = seeds.clone();

			// mask candidate seeds with predicate images
			candidates.copyTo(seeds, imgPred1 & imgPred2);

			aia::imshow("Growing", seeds, false, scaling_factor);
			cv::waitKey(10);
		}
		while(cv::countNonZero(seeds-seeds_prev) > 0);

		aia::imshow("Result", seeds, true, scaling_factor);
		//cv::imwrite("C:/work/growing.png", seeds);

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

