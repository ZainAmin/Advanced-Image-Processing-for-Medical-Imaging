// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
	cv::Mat imstdev(cv::Mat img, int kernel_size)
	{
		// making all calculations in floating-point now on
		cv::Mat img_f;
		img.convertTo(img_f, CV_32F);

		cv::Mat averaging_kernel(kernel_size, kernel_size, CV_32F, cv::Scalar(1.0/(kernel_size*kernel_size)));
		
		cv::Mat img_sq;
		cv::pow(img_f, 2.0, img_sq);
		cv::Mat avg_img_sq;
		cv::filter2D(img_sq, avg_img_sq, CV_32F, averaging_kernel);


		cv::Mat avg_img;
		cv::filter2D(img_f, avg_img, CV_32F, averaging_kernel);
		cv::Mat sq_avg_img;
		cv::pow(avg_img, 2.0, sq_avg_img);

		cv::Mat stdev_img;
		cv::sqrt(avg_img_sq-sq_avg_img, stdev_img);

		cv::normalize(stdev_img, stdev_img, 0, 255, cv::NORM_MINMAX);
		stdev_img.convertTo(stdev_img, CV_8U);

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

		// seeds image
		cv::Mat seeds;
		cv::threshold(img, seeds, 250, 255, cv::THRESH_BINARY);
		aia::imshow("Seeds image", seeds, true, 0.7);

		// predicate image 1
		cv::Mat pred1;
		cv::threshold(img, pred1, 120, 255, cv::THRESH_BINARY);
		aia::imshow("Predicate 1 image", pred1, true, 0.7);

		// predicate image 2
		/*cv::Mat pred2;
		cv::morphologyEx(pred1, pred2, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)));
		pred2 = 255 - pred2;
		aia::imshow("Predicate 2 image", pred2, true, 0.7);*/

		// alternative predicate image 2
		cv::Mat pred2;
		//aia::imshow("Stdev image", imstdev(img, 7), true, 0.7);
		cv::threshold(imstdev(img, 7), pred2, 30, 255, cv::THRESH_BINARY);
		aia::imshow("Predicate 2 image", pred2, true, 0.7);

		cv::Mat pred = pred1 & pred2;
		aia::imshow("Predicate image", pred, true, 0.7);

		// region growing
		cv::Mat seeds_prev;
		do 
		{
			seeds_prev = seeds.clone();
			cv::dilate(seeds, seeds, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			seeds = seeds_prev + (seeds - seeds_prev) & pred;

			aia::imshow("Growing", seeds, false, 0.7);
			cv::waitKey(10);
		} while ( cv::countNonZero(seeds-seeds_prev) );
		aia::imshow("Result", seeds, true, 0.7);

		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_seg.png", seeds);

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
