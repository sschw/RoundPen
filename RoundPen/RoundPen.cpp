#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void print_single_channels(const Mat& frame) {
	Mat ch1, ch2, ch3;
	// "channels" is a vector of 3 Mat arrays:
	vector<Mat> channels(3);
	// split img:
	split(frame, channels);
	// get the channels (dont forget they follow BGR order in OpenCV)
	ch1 = channels[0];
	ch2 = channels[1];
	ch3 = channels[2];
	imshow("c1", channels[0]);
	imshow("c2", channels[1]);
	imshow("c3", channels[2]);
}

int main(int argn, char** argv)
{
	cv::String path;
	if (argn == 1) {
		// Use default test video with 1 frame.
		path = "./RoundPen.mp4";
	}
	else {
		// Read filename from cmd.
		path = argv[1];
	}
	
	uint8_t colorBuffer[124];
	int colorBufferValues = argn - 2;
	if (colorBufferValues <= 0) {
		// Use some default test values.
		colorBufferValues = 7;
		colorBuffer[0] = 0;
		colorBuffer[1] = 54;
		colorBuffer[2] = 27;
		colorBuffer[3] = 98;
		colorBuffer[4] = 130;
		colorBuffer[5] = 81;
		colorBuffer[6] = 125;
	}
	else {
		// Read integers from cmd.

		// NOTE: H Value in OpenCV is [0-179]
		for (int i = 0; i < colorBufferValues; i++) {
			colorBuffer[i] = atoi(argv[i + 2]);
		}
	}

	VideoCapture cap;
	Mat frame;
	Mat frame_thresh;

	cap.open(path);
	if (!cap.isOpened()) {
		cerr << "Error opening video" << endl;
		cerr << "Call the command with a valid video file as first parameter" << endl;
		return -1;
	}

	while (cap.read(frame)) {
		imshow("img", frame);
		cvtColor(frame, frame, cv::COLOR_BGR2HSV);
		for (int i = 0; i < colorBufferValues; i++) {
			uint8_t lowerH = colorBuffer[i] - 2;
			uint8_t higherH = colorBuffer[i] + 2;
			if (lowerH > 179) lowerH -= 256 - 180;
			if (higherH > 179) lowerH -= 180;
			inRange(frame, cv::Scalar(lowerH, 0, 0), cv::Scalar(higherH, 255, 255), frame_thresh);
			imshow("Current frame color threshold " + to_string(i), frame_thresh);
		}
	}
	waitKey(0);
	return 0;
}