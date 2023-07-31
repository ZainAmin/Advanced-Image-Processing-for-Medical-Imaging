// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	cv::Mat img;

	// gradient-based edge detection params
	int sigmaX10_grad = 10;
	int threshold_grad = 33;

	// LoG ...
	int sigmaX10_LoG = 10;
	int threshold_LoG = 1;

	// Canny ...
	int sigmaX10_Canny = 10;
	int threshold_L = 10;
}

void gradientEdgeDetection(int pos, void* userdata) 
{
	//printf("sigmaX10 = %d, threshold = %d\n", sigmaX10, threshold);
	if(sigmaX10_grad <= 0)
		return;

	cv::Mat img_copy = img.clone();
	cv::GaussianBlur(img_copy, img_copy, cv::Size(0,0), sigmaX10_grad/10.0, sigmaX10_grad/10.0);
	cv::Mat dy, dx;
	cv::Sobel(img_copy, dy, CV_32F, 0, 1);
	cv::Sobel(img_copy, dx, CV_32F, 1, 0);
	cv::Mat mag;
	cv::magnitude(dx, dy, mag);
	double minVal, maxVal;
	cv::minMaxLoc(mag, &minVal, &maxVal);
	cv::threshold(mag, mag, (threshold_grad/100.0)*maxVal, 255, CV_THRESH_BINARY);

	cv::imshow("Gradient-based edge detection", mag);
}

void LoGEdgeDetection(int pos, void* userdata) 
{
	//printf("sigmaX10 = %d, threshold = %d\n", sigmaX10, threshold);
	if(sigmaX10_LoG <= 0)
		return;

	cv::Mat img_copy = img.clone();
	int n = 6*(ucas::round(sigmaX10_LoG/10.0))+1;
	cv::GaussianBlur(img_copy, img_copy, cv::Size(n,n), sigmaX10_LoG/10.0, sigmaX10_LoG/10.0);

	cv::Mat lapl_kernel = (cv::Mat_<float>(3, 3) <<
		1,  1,  1,
		1, -8,  1,
		1,  1,  1);
	cv::filter2D(img_copy, img_copy, CV_32F, lapl_kernel);

	cv::Mat edgeImg(img.rows, img.cols, CV_8U, cv::Scalar(0));
	for(int y=1; y<img_copy.rows-1; y++)
	{
		float *prevRow = img_copy.ptr<float>(y-1);
		float *currRow = img_copy.ptr<float>(y);
		float *nextRow = img_copy.ptr<float>(y+1);

		unsigned char* edgeImgRow = edgeImg.ptr<unsigned char>(y);

		for(int x=1; x<img_copy.cols-1; x++)
		{
			float N  = prevRow[x];
			float NE = prevRow[x+1];
			float E  = currRow[x+1];
			float SE = nextRow[x+1];
			float S  = nextRow[x];
			float SW = nextRow[x-1];
			float W  = currRow[x-1];
			float NW = prevRow[x-1];
			float CE = currRow[x];
			if(
				(N * S   < 0 && std::abs(N-S)   >= threshold_LoG) ||
				(NE * SW < 0 && std::abs(NE-SW) >= threshold_LoG) ||
				(E * W   < 0 && std::abs(E-W)   >= threshold_LoG) ||
				(NW * SE < 0 && std::abs(NW-SE) >= threshold_LoG)
			  )
				edgeImgRow[x] = 255;
		}
	}

	cv::imshow("LoG-based edge detection", edgeImg);
}

void CannyEdgeDetection(int pos, void* userdata) 
{
	//printf("sigmaX10 = %d, threshold = %d\n", sigmaX10, threshold);
	if(sigmaX10_Canny <= 0)
		return;

	cv::Mat img_copy = img.clone();
	cv::GaussianBlur(img_copy, img_copy, cv::Size(0,0), sigmaX10_Canny/10.0, sigmaX10_Canny/10.0);
	
	cv::Canny(img_copy, img_copy, threshold_L, 3*threshold_L);

	cv::imshow("Canny-based edge detection", img_copy);
}

int main() 
{
	try
	{
		// load an image where there are lines that can be detected
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");

		cv::namedWindow("Gradient-based edge detection");
		cv::createTrackbar("sigma", "Gradient-based edge detection", &sigmaX10_grad, 100, gradientEdgeDetection);
		cv::createTrackbar("threshold", "Gradient-based edge detection", &threshold_grad, 100, gradientEdgeDetection);

		cv::namedWindow("LoG-based edge detection");
		cv::createTrackbar("sigma", "LoG-based edge detection", &sigmaX10_LoG, 100, LoGEdgeDetection);
		cv::createTrackbar("threshold", "LoG-based edge detection", &threshold_LoG, 100, LoGEdgeDetection);

		cv::namedWindow("Canny-based edge detection");
		cv::createTrackbar("sigma", "Canny-based edge detection", &sigmaX10_Canny, 100, CannyEdgeDetection);
		cv::createTrackbar("threshold_L", "Canny-based edge detection", &threshold_L, 100, CannyEdgeDetection);

		gradientEdgeDetection(1, 0);
		LoGEdgeDetection(1, 0);
		CannyEdgeDetection(1, 0);

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