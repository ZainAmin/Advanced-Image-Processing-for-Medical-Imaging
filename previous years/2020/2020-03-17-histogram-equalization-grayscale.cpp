// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img);

		std::vector<int> hist = ucas::histogram(img);
		aia::imshow("Histogram", ucas::imhist(img));
		
		int L = std::pow(2, ucas::imdepth_detect(img));
		std::vector<int> transform(L);
		for(int k=0; k<L; k++)
		{
			transform[k] = 0;
			for(int j=0; j<=k; j++)
				transform[k] += hist[j];
			transform[k] = ucas::round(transform[k] * ((float(L)-1)/(img.rows*img.cols)));
		}

		for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				row[x] = transform[row[x]];
		}

		aia::imshow("Image post HE", img);
		aia::imshow("Histogram post HE", ucas::imhist(img));

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

