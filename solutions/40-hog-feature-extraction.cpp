// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

cv::Mat HOGimage(const cv::HOGDescriptor& hog, const std::vector<float>& features)
{
	cv::Mat img_hog(hog.winSize, CV_32F, cv::Scalar(0));

	int n_blocks_x = (hog.winSize.width - hog.blockSize.width) / hog.blockStride.width + 1;
	int n_blocks_y = (hog.winSize.height - hog.blockSize.height) / hog.blockStride.height + 1;
	
	int n_cells_bx = hog.blockSize.width / hog.cellSize.width;
	int n_cells_by = hog.blockSize.height / hog.cellSize.height;

	int n_bins = hog.nbins;

	int arrow_radius = 3;

	cv::Mat img_tmp(hog.winSize, CV_32F, cv::Scalar(0));
	for (int bi = 0; bi < n_blocks_y; bi++)
		for (int bj = 0; bj < n_blocks_x; bj++)
			for (int ci = 0; ci < n_cells_by; ci++)
				for (int cj = 0; cj < n_cells_bx; cj++)
				{
					cv::Mat cell_roi = img_tmp(cv::Rect(bj * hog.blockStride.width + cj * hog.cellSize.width, bi * hog.blockStride.height + ci * hog.cellSize.height, hog.cellSize.width, hog.cellSize.height));
					
					int hist_start = bj * n_blocks_y * n_bins * n_cells_bx * n_cells_by + 
						             bi * n_bins * n_cells_bx * n_cells_by + 
						             cj * n_cells_by * n_bins +
						             ci * n_bins;	// HOG layout is column-major for both blocks and cells

					float dominant_weight = features[hist_start];
					int dominant_angle = 0;
					for (int bin = hist_start + 1; bin < hist_start + n_bins; bin++)
					{
						if (features[bin] > dominant_weight)
						{
							dominant_weight = features[bin];
							dominant_angle = bin - hist_start;
						}
					}

					float angle = (dominant_angle / float(n_bins)) * ucas::PI + ucas::PI/2;
					img_tmp.setTo(cv::Scalar(0));
					cv::line(cell_roi,
						cv::Point2f(hog.cellSize.width / 2.f, hog.cellSize.height / 2.f) - cv::Point2f(arrow_radius * cos(angle), arrow_radius * sin(angle)),
						cv::Point2f(hog.cellSize.width / 2.f, hog.cellSize.height / 2.f) + cv::Point2f(arrow_radius * cos(angle), arrow_radius * sin(angle)), cv::Scalar(dominant_weight), 1, cv::LINE_AA);
					img_hog += img_tmp;
				}

	// normalize to [0,1], OpenCV can visualize float images if they are in this range 
	cv::normalize(img_hog, img_hog, 0, 1, cv::NORM_MINMAX);

	return img_hog;
}

int main()
{
	// HOG parameters
	int n_bins = 9;
	int cell_size = 8;
	int block_size = 5 * cell_size;	// affects normalization locality (the larger, the more data are gathered for normalization)

	// Load an image
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/girl.png");
	//cv::Mat img(600, 800, CV_8U, cv::Scalar(0));
	//cv::rectangle(img, cv::Rect(200, 200, 200, 200), cv::Scalar(255), cv::FILLED);

	// resize the image so that it can be divided by cell_size
	cv::resize(img, img, cv::Size((img.cols / cell_size) * cell_size, (img.rows / cell_size) * cell_size));

	// create HOG descriptor with:
	//   winSize   = image size since we are NOT applying object detection within the image
	//               instead we want to describe the image with one HOG-based feature vector
	cv::HOGDescriptor hog(
		cv::Size(img.cols, img.rows), 
		cv::Size(block_size, block_size),
		cv::Size(cell_size, cell_size),
		cv::Size(cell_size, cell_size), n_bins);

	// compute HOG descriptors
	ucas::Timer timer;
	std::vector<float> hog_features;
	hog.compute(img, hog_features);
	printf("%zd HOG features calculated in %.3f seconds\n", hog_features.size(), timer.elapsed<float>());

	// display HOG image
	aia::imshow("Original image", img);
	cv::Mat hog_img = HOGimage(hog, hog_features);
	aia::imshow("HOG image", hog_img);

	cv::normalize(hog_img, hog_img, 0, 255, cv::NORM_MINMAX);
	hog_img.convertTo(hog_img, CV_8U);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/girl_hog.png", hog_img);

	return EXIT_SUCCESS;
}
