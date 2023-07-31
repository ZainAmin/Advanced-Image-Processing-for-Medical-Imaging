// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace eiid
{
	// variabili globali
	cv::Mat lavagna;
	cv::Scalar colore_matita;
	bool lbutton_pressed = false;
	bool rbutton_pressed = false;
	int brush_size = 10;

	std::vector <cv::Scalar> tavolozza(3);

	void aMouseCallback(int event, int x, int y, int, void* userdata)
	{
		if(event == cv::EVENT_LBUTTONDOWN)
			lbutton_pressed = true;
		else if(event == cv::EVENT_LBUTTONUP)
			lbutton_pressed = false;
		if(event == cv::EVENT_RBUTTONDOWN)
			rbutton_pressed = true;
		else if(event == cv::EVENT_RBUTTONUP)
			rbutton_pressed = false;
		else if(event == cv::EVENT_MBUTTONDOWN)
			printf("ciao\n");

		if(lbutton_pressed)
			cv::circle(lavagna, cv::Point(x,y), brush_size, colore_matita, CV_FILLED);
		else if(rbutton_pressed)
			cv::circle(lavagna, cv::Point(x,y), brush_size, cv::Scalar(255,255,255), CV_FILLED);

		// per esercizio:
		// implementare tavolozza con selezione del colore mediante pulsante centrale

		cv::imshow("Paint", lavagna);
	}
}


int main() 
{
	try
	{	
		eiid::lavagna = cv::Mat(1000,1000, CV_8UC(3), cv::Scalar(255, 255, 255));
		eiid::colore_matita = cv::Scalar(255,0,0);
		eiid::tavolozza[0] = cv::Scalar(255,0,0);
		eiid::tavolozza[1] = cv::Scalar(0,255,0);
		eiid::tavolozza[2] = cv::Scalar(0,0,255);

		cv::namedWindow("Paint");
		cv::setMouseCallback("Paint", eiid::aMouseCallback);
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

