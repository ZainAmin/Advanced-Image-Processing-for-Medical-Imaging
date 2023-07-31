// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/text.png", cv::IMREAD_GRAYSCALE);

	cv::threshold(img, img, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_binarized.png", img);


	std::vector <int> horizontal_projection(img.rows);
	for(int y=0; y<img.rows; y++)
	{ 
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
			horizontal_projection[y] += yRow[x];

		printf("%d\n", horizontal_projection[y]);
	}

	int heights_sum = 0;
	int rows_count = 0;

	int row_start = 0;
	for (int i = 1; i<horizontal_projection.size(); i++)
	{
		if (horizontal_projection[i] != 0 && horizontal_projection[i-1] == 0)
			row_start = i;

		if (horizontal_projection[i] == 0 && horizontal_projection[i-1] != 0)
		{
			rows_count++;
			heights_sum = heights_sum + i-row_start;
		}
	}

	printf("rows_count = %d\n", rows_count);
	float avg_row_height = float(heights_sum) / rows_count;
	printf("rows average height = %.0f\n", avg_row_height);

	int vertical_SE_height = 0.8 * avg_row_height - 2;

	cv::Mat vertical_SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vertical_SE_height));
	cv::Mat img_eroded;
	cv::erode(img, img_eroded, vertical_SE);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_seeds.png", img_eroded);

	cv::Mat marker_cur = img_eroded;
	cv::Mat marker_prev;
	cv::Mat mask = img;
	do 
	{
		marker_prev = marker_cur.clone();

		cv::dilate(marker_cur, marker_cur, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
		marker_cur = marker_cur & mask;

		cv::imshow("Reconstruction in progress", marker_cur);
		cv::waitKey(100);

	} while (cv::countNonZero(marker_cur - marker_prev));

	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_long_chars.png", marker_cur);


	return EXIT_SUCCESS;
}