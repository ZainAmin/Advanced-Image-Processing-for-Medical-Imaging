// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	cv::Mat img;
	cv::Mat img_stretched;
	int gamma = 10;
	std::string win_name;
	std::vector <int> hist;

	void gammaCallback(int pos, void* userdata)
	{

		float c = std::pow(255, 1-gamma/10.0);

		for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img.ptr<unsigned char>(y);
			unsigned char* rowStretched = img_stretched.ptr<unsigned char>(y);

			for(int x=0; x<img.cols; x++)
				rowStretched[x] = ucas::round(c * std::pow(row[x], gamma/10.0));
		}

		cv::imshow(win_name, img_stretched);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cv::resize(img, img, cv::Size(0,0), 0.5, 0.5, CV_INTER_LINEAR);
		
		hist = ucas::histogram(img);
		img_stretched = img.clone();

		win_name = "Gamma correction";
		cv::namedWindow(win_name);
		cv::createTrackbar("gamma", win_name, &gamma, 100, gammaCallback);
		
		gammaCallback(0, 0);
		cv::waitKey(0);

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

