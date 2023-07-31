// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


int main() 
{
	try
	{	
		/*cv::Mat img(1000, 2000, CV_8U, cv::Scalar(155));
		ucas::imshow("immagine creata", img);

		// 1° metodo: .at
		ucas::Timer timer;
		for(int y=0; y<img.rows; y++)
			for(int x=0; x<img.cols; x++)
				img.at<unsigned char>(y,x) = (float(y) / (img.rows-1))*255;
		printf("elapsed time with .at = %f milliseconds\n", timer.elapsed<float>()*1000);
		ucas::imshow("immagine con sfumature", img);

		// 2° metodo: .ptr
		timer.restart();
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* img_row = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				img_row[x] = (float(y) / (img.rows-1))*255;
		}
		printf("elapsed time with .ptr = %f milliseconds\n", timer.elapsed<float>()*1000);
		ucas::imshow("immagine con sfumature", img);*/


		cv::Mat img(500,500, CV_8UC(3), cv::Scalar(255,255,255));

		// 1° metodo
		ucas::Timer timer;
		for(int y=0; y<img.rows; y++)
			for(int x=0; x<img.cols; x++)
			{
				if(	x < img.cols/3 )
				{
					img.at<cv::Vec3b>(y,x)[0] = 0 ;		// blue
					img.at<cv::Vec3b>(y,x)[1] =	255;	// green
					img.at<cv::Vec3b>(y,x)[2] =	0;		// red
				}
				else if(x >= img.cols*2/3)
				{
					img.at<cv::Vec3b>(y,x)[0] = 0 ;		// blue
					img.at<cv::Vec3b>(y,x)[1] =	0;		// green
					img.at<cv::Vec3b>(y,x)[2] =	255;	// red
				}
			}
		printf("elapsed time with .at = %f milliseconds\n", timer.elapsed<float>()*1000);
		ucas::imshow("bandiera", img);

		// 2° metodo: .ptr
		timer.restart();
		// per esercizio
		//printf("elapsed time with .ptr = %f milliseconds\n", timer.elapsed<float>()*1000);
		//ucas::imshow("bandiera", img);

		// 3° metodo: usare le ROI
		img.colRange(0, img.cols/3).setTo(cv::Scalar(0,255,0));
		img.colRange(img.cols*2/3, img.cols-1).setTo(cv::Scalar(0,0,255));
		ucas::imshow("bandiera con ROI", img);

		// 4° metodo: usare split/merge
		std::vector<cv::Mat> img_channels;
		cv::split(img, img_channels);
		// disegno una roi (o in altro modo) con intensità pari a 255 su ciascun canale
		// per esercizio
		cv::merge(img_channels, img);
		

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

