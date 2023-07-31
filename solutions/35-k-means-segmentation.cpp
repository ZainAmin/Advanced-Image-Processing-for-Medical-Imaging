// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	try
	{
		// load image
		cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/skin_lesion.png");
		if (!input_img.data)
			throw aia::error("Cannot open image");

		// convert to float & reshape to a [3 x W*H] Mat 
		//  (so every pixel is on a row of its own)
		cv::Mat data;
		cv::Mat img = input_img.clone();
		img.convertTo(data, CV_32F);
		data = data.reshape(1, data.total());

		// do kmeans
		cv::Mat labels, centers;
		cv::kmeans(data, 2, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1.0), 3,
			cv::KMEANS_PP_CENTERS, centers);

		// replace pixel values with their center value:
		cv::Vec3f* p = data.ptr<cv::Vec3f>();
		for (size_t i = 0; i < data.rows; i++)
		{
			int center_id = labels.at<int>(i);
			p[i] = centers.at<cv::Vec3f>(center_id);
		}

		// back to 2D, and uchar:
		img = data.reshape(3, img.rows);
		img.convertTo(img, CV_8U);

		aia::imshow("K-means clustering result", img);

		// smooth boundaries
		cv::medianBlur(img, img, 7);

		// select largest foreground component
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		cv::threshold(img, img, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
		std::vector< std::vector <cv::Point> > objects;
		cv::findContours(img, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		int max_area_idx = -1;
		double max_area = -1;
		for (int i = 0; i < objects.size(); i++)
		{
			double A = cv::contourArea(objects[i]);
			if (A > max_area)
			{
				max_area = A;
				max_area_idx = i;
			}
		}

		// display final result
		cv::drawContours(input_img, objects, -1, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		cv::drawContours(input_img, objects, max_area_idx, cv::Scalar(22, 14, 177), 2, cv::LINE_AA);

		aia::imshow("Segmentation result", input_img);

		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/skin_lesion_segmented.png", input_img);

		return EXIT_SUCCESS;
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}
