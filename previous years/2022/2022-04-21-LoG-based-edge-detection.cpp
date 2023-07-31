// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "Gradient edge detection demo";
	int sigma_x10 = 10;
	int zero_cross_thresh_perc = 10;
	int grad_mag_thresh_perc = 10;

	void improvedLoGEdgeDetection(int pos, void* userdata)
	{
		// first derivative
		cv::Mat dx, dy;
		cv::Sobel(img, dx, CV_32F, 1, 0);
		cv::Sobel(img, dy, CV_32F, 0, 1);
		cv::Mat mag;
		cv::magnitude(dx, dy, mag);
		double minVG, maxVG;
		cv::minMaxLoc(mag, &minVG, &maxVG);

		// LoG
		cv::Mat img_LoG;
		float sigma = sigma_x10 / 10.0f;
		img_LoG = img.clone();
		img_LoG.convertTo(img_LoG, CV_32F);
		int k = 6 * sigma;
		if (k % 2 == 0)
			k += 1;
		cv::GaussianBlur(img_LoG, img_LoG, cv::Size(k, k), sigma, sigma);
		cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) <<
			1,  1,  1,
			1, -8,  1,
			1,  1,  1);
		cv::filter2D(img_LoG, img_LoG, CV_32F, laplacian_kernel);
		double minVL, maxVL;
		cv::minMaxLoc(img_LoG, &minVL, &maxVL);
		float zero_crossing_thresh = (zero_cross_thresh_perc / 100.0f) * maxVL;
		float grad_mag_thresh = (grad_mag_thresh_perc / 100.0f) * maxVG;
		cv::Mat output(img.rows, img.cols, CV_8U, cv::Scalar(0));
		for (int y = 1; y < img_LoG.rows - 1; y++)
		{
			float* prev_row = img_LoG.ptr<float>(y - 1);
			float* curr_row = img_LoG.ptr<float>(y);
			float* next_row = img_LoG.ptr<float>(y + 1);

			float* mag_row = mag.ptr<float>(y);

			unsigned char* out_row = output.ptr<unsigned char>(y);

			for (int x = 1; x < img_LoG.cols - 1; x++)
			{
				float N  = prev_row[x];
				float NE = prev_row[x + 1];
				float E  = curr_row[x + 1];
				float SE = next_row[x + 1];
				float S  = next_row[x];
				float SW = next_row[x - 1];
				float W  = curr_row[x - 1];
				float NW = prev_row[x - 1];

				if (mag_row[x] > grad_mag_thresh)
				{
					if ((N * S   < 0 && std::abs(N - S)   > zero_crossing_thresh) ||
						(NE * SW < 0 && std::abs(NE - SW) > zero_crossing_thresh) ||
						(E * W   < 0 && std::abs(E - W)   > zero_crossing_thresh) ||
						(SE * NW < 0 && std::abs(SE - NW) > zero_crossing_thresh) )
						out_row[x] = 255;
				}
				//else
				//	out_row[x] = 0;			
			}
		}

		cv::imshow(win_name, output);
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/brain_ct.jpeg", cv::IMREAD_GRAYSCALE);


	cv::namedWindow(win_name);
	cv::createTrackbar("sigma", win_name, &sigma_x10, 100, improvedLoGEdgeDetection);
	cv::createTrackbar("zero cross thres", win_name, &zero_cross_thresh_perc, 100, improvedLoGEdgeDetection);
	cv::createTrackbar("grad mag thres", win_name, &grad_mag_thresh_perc, 100, improvedLoGEdgeDetection);
	improvedLoGEdgeDetection(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}