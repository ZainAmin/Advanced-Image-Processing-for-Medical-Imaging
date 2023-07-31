// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

namespace
{
	int intensity_diff_thresh = 2;

	bool intensityDifferenceCriterion(unsigned char a, unsigned char b)
	{
		return std::abs(int(a)-int(b)) <= intensity_diff_thresh;
	}

	cv::Mat imStdev(cv::Mat img, int filter_size)
	{
		cv::Mat img_avg;
		cv::boxFilter(img, img_avg, CV_32F, cv::Size(filter_size, filter_size));
		cv::Mat img_avg_sq;
		cv::pow(img_avg, 2, img_avg_sq);

		cv::Mat img_sq;
		img.convertTo(img_sq, CV_32F);
		cv::pow(img_sq, 2, img_sq);
		cv::Mat img_sq_avg;
		cv::boxFilter(img_sq, img_sq_avg, CV_32F, cv::Size(filter_size, filter_size));

		cv::Mat stdev;
		cv::sqrt(img_sq_avg-img_avg_sq, stdev);

		cv::normalize(stdev, stdev, 0, 255, cv::NORM_MINMAX);
		stdev.convertTo(stdev, CV_8U);

		return stdev;
	}
}

int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");

		// define seed image
		cv::Mat seed;
		cv::threshold(img, seed, 250, 255, CV_THRESH_BINARY);


		// DEBUGGING: check the seed image
		/*cv::Mat seed_img_check = img.clone();
		cv::cvtColor(seed_img_check, seed_img_check, cv::COLOR_GRAY2BGR);
		seed_img_check.setTo(cv::Scalar(0,0,255), seed);
		aia::imshow("Seed image check", seed_img_check);*/

		// REGION-GROWING based on difference in intensity
		//cv::Mat seed_prev = seed.clone();
		//do 
		//{
		//	seed.copyTo(seed_prev);

		//	for(int y=1; y<seed.rows-1; y++)
		//	{
		//		unsigned char* imgPrevRow = img.ptr<unsigned char>(y-1);
		//		unsigned char* imgCurrRow = img.ptr<unsigned char>(y);
		//		unsigned char* imgNextRow = img.ptr<unsigned char>(y+1);

		//		unsigned char* seedPrevRow = seed.ptr<unsigned char>(y-1);
		//		unsigned char* seedCurrRow = seed.ptr<unsigned char>(y);
		//		unsigned char* seedNextRow = seed.ptr<unsigned char>(y+1);

		//		unsigned char* seedRow = seed_prev.ptr<unsigned char>(y);
		//		
		//		for(int x=1; x<seed.cols-1; x++)
		//		{
		//			// only seed pixels can grow
		//			if(seedRow[x])
		//			{
		//				if(     intensityDifferenceCriterion(imgCurrRow[x], imgPrevRow[x-1]))
		//					seedPrevRow[x-1] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgPrevRow[x]))
		//					seedPrevRow[x] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgPrevRow[x+1]))
		//					seedPrevRow[x+1] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgCurrRow[x-1]))
		//					seedCurrRow[x-1] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgCurrRow[x+1]))
		//					seedCurrRow[x+1] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgNextRow[x+1]))
		//					seedNextRow[x+1] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgNextRow[x]))
		//					seedNextRow[x] = 255;
		//				if(intensityDifferenceCriterion(imgCurrRow[x], imgNextRow[x-1]))
		//					seedNextRow[x-1] = 255;
		//			}
		//		}
		//	}

		//	cv::imshow("Region Growing", seed);
		//	cv::waitKey(0);

		//} while (cv::countNonZero(seed-seed_prev));

		//aia::imshow("Region Growing result", seed);


		// REGION GROWING based on predicate images
		
		// predicate image #1
		cv::Mat pred1;
		cv::threshold(img, pred1, 100, 255, CV_THRESH_BINARY);

		// predicate image #2
		cv::Mat pred2;
		cv::Mat imstdev = imStdev(img, 7);
		cv::threshold(imstdev, pred2, 20, 255, CV_THRESH_BINARY);

		cv::Mat seed_prev = seed.clone();
		do 
		{
			seed.copyTo(seed_prev);

			cv::Mat candidates;
			cv::dilate(seed, candidates, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			candidates -= seed;

			seed += candidates & pred1 & pred2;

			cv::imshow("Region Growing", seed);
			cv::waitKey(50);

		} while (cv::countNonZero(seed-seed_prev));

		aia::imshow("Region Growing result", seed);

		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray_seg.png", seed);
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
