// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	// gradient parameters
	int sigmaX10 = 1;
	int thresh = 50;

	// LoG parameters
	int LoG_sigmaX10 = 1;
	int LoG_thresh = 50;

	// other stuff
	std::string win_name_1 = "Gradient-based edge detection";
	std::string win_name_2 = "LoG edge detection";
	std::string win_name_3 = "Canny edge detection";
	cv::Mat img;

	void gradientEdgeDetectionCallback(int pos, void* userdata)
	{
		if (sigmaX10 == 0)
			return;

		// preprocessing: gaussian smoothing
		cv::Mat img_preprocessed;
		cv::GaussianBlur(img, img_preprocessed, cv::Size(0, 0), sigmaX10 / 10.0);

		// sobel derivatives
		cv::Mat dx, dy;
		cv::Sobel(img_preprocessed, dx, CV_32F, 1, 0);
		cv::Sobel(img_preprocessed, dy, CV_32F, 0, 1);
		cv::Mat mag;
		cv::magnitude(dx, dy, mag);
		cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
		mag.convertTo(mag, CV_8U);

		double minV, maxV;
		cv::minMaxLoc(mag, &minV, &maxV);
		int T = ucas::round((thresh / 100.0) * maxV);
		cv::threshold(mag, mag, T, 255, cv::THRESH_BINARY);

		cv::imshow(win_name_1, mag);
	}

	void LoGDetectionCallback(int pos, void* userdata)
	{
		if (LoG_sigmaX10 == 0)
			return;

		// gaussian convolution
		cv::Mat img_gaussian;
		int n = ucas::round(6 * LoG_sigmaX10 / 10.0);
		if (n % 2 == 0)
			n++;
		cv::GaussianBlur(img, img_gaussian, cv::Size(n, n), LoG_sigmaX10 / 10.0);

		// laplacian convolution
		cv::Mat laplKernel = (cv::Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);
		cv::Mat img_LoG;
		cv::filter2D(img_gaussian, img_LoG, CV_32F, laplKernel);

		// allocate result image
		cv::Mat result(img.rows, img.cols, CV_8U, cv::Scalar(0));

		// thresholded zero-crossing
		double minV, maxV;
		cv::minMaxLoc(cv::abs(img_LoG), &minV, &maxV);
		double T = (LoG_thresh / 100.0) * maxV;
		for (int y = 1; y < img_LoG.rows - 1; y++)
		{
			float* prevRow = img_LoG.ptr<float>(y - 1);
			float* currRow = img_LoG.ptr<float>(y);
			float* nextRow = img_LoG.ptr<float>(y + 1);
			unsigned char* resultRow = result.ptr<unsigned char>(y);
			for (int x = 1; x < img_LoG.cols - 1; x++)
			{
				float N = prevRow[x];
				float NE = prevRow[x + 1];
				float E = currRow[x + 1];
				float SE = nextRow[x + 1];
				float S = nextRow[x];
				float SW = nextRow[x - 1];
				float W = currRow[x - 1];
				float NW = prevRow[x - 1];
				if (N < 0 && S > 0 && std::abs(N - S) >= T)
					resultRow[x] = 255;
				else if (N > 0 && S < 0 && std::abs(N - S) >= T)
					resultRow[x] = 255;
				else if (W > 0 && E < 0 && std::abs(W - E) >= T)
					resultRow[x] = 255;
				else if (W < 0 && E > 0 && std::abs(W - E) >= T)
					resultRow[x] = 255;
				else if (NE > 0 && SW < 0 && std::abs(NE - SW) >= T)
					resultRow[x] = 255;
				else if (NE < 0 && SW > 0 && std::abs(NE - SW) >= T)
					resultRow[x] = 255;
				else if (NW > 0 && SE < 0 && std::abs(NW - SE) >= T)
					resultRow[x] = 255;
				else if (NW < 0 && SE > 0 && std::abs(NW - SE) >= T)
					resultRow[x] = 255;
			}
		}

		cv::imshow(win_name_2, result);
	}
}

int main()
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", cv::IMREAD_GRAYSCALE);
		
		// create the UIs
		cv::namedWindow(win_name_1);
		cv::createTrackbar("sigma_x10", win_name_1, &sigmaX10, 100, gradientEdgeDetectionCallback);
		cv::createTrackbar("thresh", win_name_1, &thresh, 100, gradientEdgeDetectionCallback);
		cv::namedWindow(win_name_2);
		cv::createTrackbar("sigma_x10", win_name_2, &LoG_sigmaX10, 100, LoGDetectionCallback);
		cv::createTrackbar("thresh", win_name_2, &LoG_thresh, 100, LoGDetectionCallback);

		// start the UIs
		gradientEdgeDetectionCallback(0, 0);
		LoGDetectionCallback(0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}

	return EXIT_SUCCESS;
}


