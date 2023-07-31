// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv image processing module
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>

// include file I/O library
#include <fstream>

// THE GOAL
namespace facerecognition
{
	// the model/classifier that performs face recognition
	static cv::Ptr<cv::face::LBPHFaceRecognizer> model;

	// <ID, name> table (the model works with integer IDs, but we also deal with person names!)
	static std::map <int, std::string> id2names;

	// how many samples we need per face (at training time)
	const int num_samples_per_face = 10;

	// sample (= face image) fixed height
	// all face images are resized to the same scale (this facilitates training the model)
	// the scale is fixed by resizing to the same height (92 pixels is commonly used in the literature)
	const int sample_fixed_height = 92;

	// model initialization
	void init() throw (aia::error);

	// model update / add a new face to the model and re-train
	void update() throw (aia::error);

	// single-frame processor: takes a frame in input, and returns an image with the recognized face in output
	cv::Mat process(const cv::Mat& frame);
};

int main()
{
	try
	{
		// initialize the system
		facerecognition::init();
		system("pause");

		std::string choice;
		do
		{
			system("cls");
			printf("[1] Add a new face and retrain\n");
			printf("[2] Face recognition\n");
			printf("[0] Exit\n\n");

			printf("Insert option: ");
			std::getline(std::cin, choice);

			if (choice == "1")
			{
				system("cls");
				facerecognition::update();
			}
			else if (choice == "2")
			{
				system("cls");
				printf("Face recognition in progress...\n");
				aia::processVideoStream("", facerecognition::process, "", true);
			}
		} while (choice != "0");


		return 1;
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

// model initialization
void facerecognition::init() throw (aia::error)
{
	// create LBP-H(istogram) based face recognition model
	printf("Create the model..............");
	model = cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8);	// LBP(8,1) with blocks size = (8,8)
	printf("DONE\n");

	// load model from file if exists
	printf("Load existing model...........");
	if (aia::isFile("model.yaml"))
	{
		facerecognition::model->read("model.yaml");
		printf("DONE, LBP is P=%d, R=%d, sample number = %d\n",
			facerecognition::model->getNeighbors(),
			facerecognition::model->getRadius(),
			facerecognition::model->getHistograms().size());
	}
	else
		printf("NOT FOUND\n");

	// load <ID, name> table if exists
	printf("Load existing id2name table...");
	if (aia::isFile("id2names.csv"))
	{
		std::ifstream f("id2names.csv");
		std::string line;
		while (std::getline(f, line))
		{
			std::vector <std::string> fields;
			aia::split(line, ",", fields);
			if (fields.size() != 2)
				throw aia::error(aia::strprintf("Error when parsing id2names.txt: expected 2 fields, found %d", fields.size()));
			facerecognition::id2names[aia::str2num<int>(fields[0])] = fields[1];
		}
		f.close();
		printf("DONE, unique faces (=persons) = %d\n", facerecognition::id2names.size());
	}
	else
		printf("NOT FOUND\n");
}

// model update
void facerecognition::update() throw (aia::error)
{
	// ask user name
	printf("\nInsert your full name: ");
	std::string name;
	std::getline(std::cin, name);
	printf("\nDear %s, now your face will being scanned.\n", name.c_str());
	printf("We will try to acquire %d different images of your face.\n", num_samples_per_face);
	printf("When your face is shown correctly, press \'y\' to accept or another key to reject\n\n");


	// add name to the <ID, name> table and assign unique ID (= the smaller unused integer = the size of id2names)
	int fid = facerecognition::id2names.size();
	facerecognition::id2names[fid] = name;


	// open input video stream from camera
	cv::VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
		throw aia::error("Cannot open input video stream from camera");


	// acquire 10 different images of the user's face
	printf("Start capture.................\n");
	std::vector <cv::Mat> face_samples;
	bool stop(false);
	while (!stop)
	{
		// read next frame if any
		cv::Mat frame;
		if (!capture.read(frame))
			break;

		// run face detector (internally, it uses the Viola & Jones Cascade face detector [2001] )
		std::vector < cv::Rect > faces = aia::faceDetector(frame);

		// if at least one face was found, we show the first face to user and ask him/her to accept
		if (faces.size())
		{
			cv::imshow("Accept? [y/n]", frame(faces[0]));
			int choice = cv::waitKey();
			if (choice == (int)('y'))
			{
				face_samples.push_back(frame(faces[0]).clone());
				printf("   %d camera shots remaining\n", num_samples_per_face - face_samples.size());

				// stop when we reach the desired number of samples
				if (face_samples.size() == num_samples_per_face)
					stop = true;
			}
		}
	}
	printf("DONE\n\n");
	cv::destroyWindow("Accept? [y/n]");

	// resize all face images to the same fixed scale (this facilitates training the model)
	// ...also convert to grayscale (OpenCV's LBP implementation only works on grayscale)
	for (auto& f : face_samples)
	{
		double rescale_factor = double(sample_fixed_height) / f.rows;
		cv::resize(f, f, cv::Size(0, 0), rescale_factor, rescale_factor, cv::INTER_AREA);
		cv::cvtColor(f, f, cv::COLOR_BGR2GRAY);
	}

	// update / retrain the model
	printf("Update model..................");
	std::vector<int> face_ids(face_samples.size());
	for (auto& id : face_ids)
		id = fid;
	facerecognition::model->update(face_samples, face_ids);
	printf("DONE\n");

	// save the model to disk
	printf("Save model....................");
	facerecognition::model->save("model.yaml");

	// save the <ID, name> table to disk
	std::ofstream f("id2names.csv");
	for (auto& el : facerecognition::id2names)
		f << el.first << "," << el.second << std::endl;
	f.close();
	printf("DONE\n");
}

// single-frame processor: takes a frame in input, and returns an image with the recognized face in output
cv::Mat facerecognition::process(const cv::Mat& frame) throw (aia::error)
{
	// run face detector (internally, it uses the Viola & Jones Cascade face detector [2001] )
	std::vector < cv::Rect > faces = aia::faceDetector(frame);

	// prepare output
	cv::Mat out = frame.clone();

	// recognize each face
	for (auto& face : faces)
	{
		// discard too small samples
		if (face.area() < 500)
			continue;

		// resize and convert to gray (the exact same steps we did at training time)
		cv::Mat face_img = frame(face).clone();
		double rescale_factor = double(sample_fixed_height) / face_img.rows;
		cv::resize(face_img, face_img, cv::Size(0, 0), rescale_factor, rescale_factor, cv::INTER_AREA);
		cv::cvtColor(face_img, face_img, cv::COLOR_BGR2GRAY);

		// get the prediction label from the model
		int predictedLabel = facerecognition::model->predict(face_img);

		// draw the rectangle and the face name
		cv::rectangle(out, face, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::putText(
			out,
			facerecognition::id2names[predictedLabel],
			cv::Point(face.x, face.y - 10),
			3, 1.0, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
	}

	return out;
}