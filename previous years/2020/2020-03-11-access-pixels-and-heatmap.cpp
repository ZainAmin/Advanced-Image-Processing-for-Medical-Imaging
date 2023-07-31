// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat getHeatMap(const cv::Mat & img)
{
	double minV, maxV;
	cv::minMaxLoc(img, &minV, &maxV);
	printf("min = %f, max = %f\n", minV, maxV);

	cv::Mat hsv[3];
	hsv[0] = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(90));
	hsv[1] = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(255));
	hsv[2] = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(255));

	hsv[0] = ((maxV-img)/float(maxV))*120;

	cv::Mat heatmap;
	cv::merge(hsv, 3, heatmap);
	cv::cvtColor(heatmap, heatmap, cv::COLOR_HSV2BGR);

	return heatmap;
}

int main() 
{
	try
	{
		//cv::Mat img(500, 1000, CV_8U);
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);

		printf("rows = %d, cols = %d, channels = %d, imdepth = %d\n",
			img.rows, img.cols, img.channels(), aia::bitdepth(img.depth()));

		if(!img.data)
			throw ucas::Error("Invalid image");

		// grayscale pixel access (approach 1)
		/*ucas::Timer timer;
		for(int y=0; y<img.rows; y++)
			for(int x=0; x<img.cols; x++)
				img.at<unsigned char>(y, x) = 120;
		printf("Easy/slow pixel access = %.3f\n", timer.elapsed<float>());

		// grayscale pixel access (approach 2)
		timer.restart();
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				row[x] = 120;
		}
		printf("Hard/fast pixel access = %.3f\n", timer.elapsed<float>());
		*/

		// grayshades image
		/*for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				row[x] = (y/float(img.rows-1))*255;
		}*/
		aia::imshow("Immagine", img, true, 0.5);

		// create heatmap image
		aia::imshow("Mappa di calore", getHeatMap(img), true, 0.5);

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

