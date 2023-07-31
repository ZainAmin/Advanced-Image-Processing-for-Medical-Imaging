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
	int threshold_div10 = 20;
	std::string win_name;

	void LoGEdgeDetection(int pos, void* userdata)
	{
		cv::Mat LoG_kernel = 
			(cv::Mat_<float>(5,5) <<
				0,   0, -1,  0,  0,
				0,  -1, -2, -1,  0,
				-1, -2, 16, -2, -1,
				0,  -1, -2, -1,  0,
				0,   0, -1,  0,  0
			);

		cv::Mat LoG_res;
		cv::filter2D(img, LoG_res, CV_32F, LoG_kernel);

		cv::Mat edge_img(img.rows, img.cols, CV_8U, cv::Scalar(0));

		int threshold = threshold_div10*10;

		for(int y=1; y < LoG_res.rows-1; y++)
		{
			float *prevRow = LoG_res.ptr<float>(y-1);
			float *currRow = LoG_res.ptr<float>(y);
			float *nextRow = LoG_res.ptr<float>(y+1);

			unsigned char* edgeImgRow = edge_img.ptr<unsigned char>(y);

			for(int x=1; x < LoG_res.cols-1; x++)
			{
				float N  = prevRow[x];
				float NE = prevRow[x+1];
				float E  = currRow[x+1];
				float SE = nextRow[x+1];
				float S  = nextRow[x];
				float SW = nextRow[x-1];
				float W  = currRow[x-1];
				float NW = prevRow[x-1];

				if(     N > 0 && S < 0 && std::abs(N-S)>threshold)
					edgeImgRow[x] = 255;
				else if(S > 0 && N < 0 && std::abs(N-S)>threshold)
					edgeImgRow[x] = 255;
				else if(W > 0 && E < 0 && std::abs(W-E)>threshold)
					edgeImgRow[x] = 255;
				else if(E > 0 && W < 0 && std::abs(W-E)>threshold)
					edgeImgRow[x] = 255;
				else if(NE > 0 && SW < 0 && std::abs(NE-SW)>threshold)
					edgeImgRow[x] = 255;
				else if(SW > 0 && NE < 0 && std::abs(NE-SW)>threshold)
					edgeImgRow[x] = 255;
				else if(SE > 0 && NW < 0 && std::abs(SW-NW)>threshold)
					edgeImgRow[x] = 255;
				else if(NW > 0 && SW < 0 && std::abs(SW-NW)>threshold)
					edgeImgRow[x] = 255;
				else
					;
			}
		}
		
		cv::imshow(win_name, edge_img);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/child.png");
		aia::imshow("Image", img);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

		win_name = "LoG-based edge detection";
		cv::namedWindow(win_name);
		cv::createTrackbar("threshold", win_name, &threshold_div10, 100, LoGEdgeDetection);

		LoGEdgeDetection(0, 0);
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

