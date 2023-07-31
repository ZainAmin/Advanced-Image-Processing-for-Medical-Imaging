// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{
		cv::Mat img(500, 500, CV_8U, cv::Scalar(0));
		cv::rectangle(img, cv::Rect(150, 150, 250, 250), cv::Scalar(255), 1);
		//cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree.tif", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img);

		// create marker with METHOD 1 (pixel access)
		cv::Mat marker1 = img.clone();
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* yRowMarker = marker1.ptr<unsigned char>(y);
			unsigned char* yRowImg    = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				if(x == 0 || x == img.cols-1 || y == 0 || y == img.rows-1)
					yRowMarker[x] = 255 - yRowImg[x];
				else
					yRowMarker[x] = 0;
		}

		// create marker with METHOD 2 (masking)
		cv::Mat mask = img.clone();
		mask.setTo(cv::Scalar(0));
		cv::rectangle(mask, cv::Rect(0, 0, img.cols, img.rows), cv::Scalar(255), 1);
		cv::Mat marker2(img.rows, img.cols, CV_8U, cv::Scalar(0));
		img = 255 - img;
		img.copyTo(marker2, mask);
		img = 255 - img;
		aia::imshow("Marker 1", marker1);
		aia::imshow("Marker 2", marker2);
		printf("difference between markers = %d\n", cv::countNonZero(cv::abs(marker1-marker2)));

		cv::Mat rec_mask = 255 - img;
		cv::Mat marker = marker1;
		cv::Mat marker_prev;
		do 
		{
			marker_prev = marker.clone();

			// geodesic dilation = dilation + intersection with mask
			// CROSS SE for 4-adjacency-based reconstruction
			cv::dilate(marker, marker, 
				cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
			marker = marker & rec_mask;

			cv::waitKey(50);
			cv::imshow("Reconstruction in progress", marker);

		} while ( cv::countNonZero(marker - marker_prev));

		aia::imshow("After reconstruction", 255-marker);

		return EXIT_SUCCESS;
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