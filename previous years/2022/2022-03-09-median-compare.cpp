// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", cv::IMREAD_GRAYSCALE);
	cv::Mat out(img.rows, img.cols, CV_8U, cv::Scalar(0));
	int k = 7;
	int h = k / 2;
	int m = (k * k) / 2;

	ucas::Timer timer;

	// filter buffer preallocation
	std::vector<unsigned char> filter_buf(k * k);
	for (int y = h; y < img.rows - h; y++)
	{
		unsigned char* out_yRow = out.ptr<unsigned char>(y);

		for (int x = h; x < img.cols - h; x++)
		{
			cv::Mat filter(img, cv::Rect(x-h, y-h, k, k));
			int i = 0;
			for (int y = 0; y < filter.rows; y++)
			{
				unsigned char* filter_yRow = filter.ptr<unsigned char>(y);
				for (int x = 0; x < filter.cols; x++)
					filter_buf[i++] = filter_yRow[x];
			}

			std::nth_element(filter_buf.begin(), filter_buf.begin() + m, filter_buf.end());
			out_yRow[x] = filter_buf[m];
		}
	}
	printf("Elapsed time (custom median filter) = %.3f seconds\n", timer.elapsed<float>());

	timer.restart();
	cv::Mat dst;
	cv::medianBlur(img, dst, k);
	printf("Elapsed time (OpenCV median filter) = %.3f seconds\n", timer.elapsed<float>());

	cv::imshow("Image median-filtered (custom)", out);
	aia::imshow("Image median-filtered (OpenCV)", dst);

	return EXIT_SUCCESS;
}