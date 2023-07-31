// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
	cv::Mat imStDev(const cv::Mat & img, int ksize)
	{
		// making all calculations in floating-point now on
		cv::Mat img_f;
		img.convertTo(img_f, CV_32F);

		// generate averaging kernel
		cv::Mat avg_kernel = cv::Mat::ones(ksize, ksize, CV_32F);
		avg_kernel /= ksize*ksize;

		cv::Mat img_sq, img_sq_avg, img_avg, img_avg_sq;

		// squared average image
		cv::filter2D(img_f, img_avg, CV_32F, avg_kernel);
		cv::pow(img_avg, 2, img_avg_sq);

		// average squared image
		cv::pow(img_f, 2, img_sq);
		cv::filter2D(img_sq, img_sq_avg, CV_32F, avg_kernel);

		// compute standard deviation
		cv::Mat stdev_img;
		cv::sqrt(img_sq_avg-img_avg_sq, stdev_img);

		return stdev_img;
	}
}

int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");
		float scaling_factor = 0.5;
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", img);


		// parameters
		int thresholdSeed = 230;
		int thresholdPred1 = 110;
		int thresholdPred2 = 20;

		// generate seeds image
		cv::Mat seeds;
		cv::threshold(img, seeds, thresholdSeed, 255, CV_THRESH_BINARY);
		aia::imshow("Seeds", seeds, true, scaling_factor);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_seeds.png", seeds);

		// generate predicate 1 image
		cv::Mat img_pred_1;
		cv::threshold(img, img_pred_1, thresholdPred1, 255, CV_THRESH_BINARY);
		aia::imshow("Predicate image 1", img_pred_1, true, scaling_factor);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_pred1.png", img_pred_1);

		// generate predicate 2 image
		cv::Mat stdev_img = imStDev(img, 5);
		cv::normalize(stdev_img, stdev_img, 0, 255, cv::NORM_MINMAX);
		stdev_img.convertTo(stdev_img, CV_8U);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_stdev.png", stdev_img);
		cv::Mat img_pred_2;
		cv::threshold(stdev_img, img_pred_2, thresholdPred2, 255, CV_THRESH_BINARY);
		aia::imshow("Predicate image 2", img_pred_2, true, scaling_factor);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_pred2.png", img_pred_2);

		// overall predicate image
		cv::Mat img_pred = img_pred_1 & img_pred_2;

		// region growing
		cv::Mat seeds_prev;
		do
		{
			seeds_prev = seeds.clone();
			cv::dilate(seeds, seeds, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			cv::Mat candidates = seeds - seeds_prev;
			seeds = seeds_prev + candidates & img_pred;

			aia::imshow("Region Growing", seeds, false, scaling_factor);
			cv::waitKey(50);
		}
		while( cv::countNonZero(seeds - seeds_prev) > 0);
		aia::imshow("Region Growing (result)", seeds, true, scaling_factor);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_growing.png", seeds);


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
