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
	int aPerc = 5;
	int bPerc = 95;
	std::string win_name;
	std::vector <int> hist;

	int percentile(const std::vector<int> & _hist, int rows, int cols, int perc)
	{
		int sum = 0;
		int k = 0;
		for(; k<_hist.size(); k++)
		{
			sum += _hist[k];
			if(sum >= (perc*rows*cols)/100)
				break;
		}
		return k;
	}

	void linearStretchingCallback(int pos, void* userdata)
	{
		int a = percentile(hist, img.rows, img.cols, aPerc);
		int b = percentile(hist, img.rows, img.cols, bPerc);

		if(b < a)
			return;
		//printf("a = %d\nb=%d\n", a, b);

		for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img.ptr<unsigned char>(y);
			unsigned char* rowStretched = img_stretched.ptr<unsigned char>(y);

			for(int x=0; x<img.cols; x++)
			{
				if(row[x] < a)
					rowStretched[x] = 0;
				else if(row[x] >= a && row[x] < b)
					rowStretched[x] = (row[x] - a)*(255/(b-a));
				else
					rowStretched[x] = 255;
			}
		}

		cv::imshow(win_name, img_stretched);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", CV_LOAD_IMAGE_GRAYSCALE);
		hist = ucas::histogram(img);
		img_stretched = img.clone();

		win_name = "Piecewise Linear Stretching (N=3)";
		cv::namedWindow(win_name);
		cv::createTrackbar("a (perc)", win_name, &aPerc, 100, linearStretchingCallback);
		cv::createTrackbar("b (perc)", win_name, &bPerc, 100, linearStretchingCallback);

		linearStretchingCallback(0, 0);
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

