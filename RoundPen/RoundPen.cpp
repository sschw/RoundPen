#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

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

void cut_roundPen(Mat& input_hsv, Mat& output_hsv, Mat& mask, Mat& downscaled, vector<vector<Point>>& contours, vector<vector<Point>>& hull, vector<Vec4i> hierarchy, bool show_imgs = false) {
	int contourID = -1;
	double area, arclength, circularity;
	double prevCircularity = 0;

	//Point2f roundPenCenter;
	//float roundPenRadius;

	//GaussianBlur(frame, frame_gaussian, Size(45, 45), 0);
	resize(input_hsv, downscaled, Size(), 0.2, 0.2);
	inRange(downscaled, Scalar(5, 3, 0), Scalar(25, 90, 255), downscaled);

	findContours(downscaled, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if (area > 2000) {
			arclength = arcLength(contours[i], true);
			circularity = 4 * CV_PI * area / (arclength * arclength);
			if (circularity > prevCircularity) {
				prevCircularity = circularity;
				contourID = i;
			}
		}
	}

	mask = Mat(input_hsv.rows, input_hsv.cols, CV_8UC1, Scalar(0));
	if (contourID >= 0) {
		if (show_imgs == true) {
			drawContours(downscaled, contours, -1, Scalar(128), -1);
			drawContours(downscaled, contours, contourID, Scalar(255), -1);
			imshow("tmp", downscaled);
		}
		//minEnclosingCircle(contours[contourID], roundPenCenter, roundPenRadius);
		//cout << roundPenCenter.x << ", " << roundPenCenter.y << ": " << roundPenRadius << endl;
		//roundPenRadius *= 5;
		//roundPenCenter.x *= 5;
		//roundPenCenter.y *= 5;
		//cout << roundPenCenter.x << ", " << roundPenCenter.y << ": " << roundPenRadius << endl;

		//circle(mask, roundPenCenter, (int)roundPenRadius, Scalar(255), -1);
		//for (int i = 0; i < contours[contourID].size(); i++) {
		//	contours[contourID][i].x *= 5;
		//	contours[contourID][i].y *= 5;
		//}
		//drawContours(mask, contours, contourID, Scalar(255), -1);
		convexHull(contours[contourID], hull[0]);
		//drawContours(mask, hull, 0, Scalar(255), -1);
		fillConvexPoly(mask, Mat(hull[0]) * 5, Scalar(255));
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * 6 + 1, 2 * 6 + 1),
			Point(6, 6));
		erode(mask, mask, element);
		if(show_imgs)
			imshow("roundpen mask", mask);
	}
	input_hsv.copyTo(output_hsv, mask);
}

int main(int argn, char** argv)
{
	String path;
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
		colorBufferValues = 10;
		// Colors from https://ae01.alicdn.com/kf/H01c3b911feb44d2381d8f5dafd851cce9/50-25-10-pc-1Pack-Farbige-Ping-Pong-B-lle-40mm-2-4g-Unterhaltung-Tischtennis-B.jpg
		colorBuffer[0] = 153;
		colorBuffer[1] = 178;
		colorBuffer[2] = 20;
		colorBuffer[3] = 103;
		colorBuffer[4] = 82;
		colorBuffer[5] = 116;
		colorBuffer[6] = 27;
		colorBuffer[7] = 99;
		colorBuffer[8] = 156;
		colorBuffer[9] = 72;
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
	Mat mask;
	Mat downscaled;
	Mat frame_only_roundpen;
	Mat frame_thresh;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> hull(1);

	cap.open(path);
	if (!cap.isOpened()) {
		cerr << "Error opening video" << endl;
		cerr << "Call the command with a valid video file as first parameter" << endl;
		return -1;
	}

	while (cap.read(frame)) {
		
		imshow("img", frame);
		cvtColor(frame, frame, COLOR_BGR2HSV);
		// Find RoundPen
		cut_roundPen(frame, frame_only_roundpen, mask, downscaled, contours, hull, hierarchy);
		
		imshow("img_cut", frame_only_roundpen);

		for (int i = 0; i < colorBufferValues; i++) {
			uint8_t lowerH = colorBuffer[i] - 1;
			uint8_t higherH = colorBuffer[i] + 1;
			if (lowerH > 179) lowerH -= 256 - 180;
			if (higherH > 179) lowerH -= 180;
			inRange(frame_only_roundpen, Scalar(lowerH, 90, 15), Scalar(higherH, 255, 240), frame_thresh);
			imshow("Current frame color threshold " + to_string(i), frame_thresh);
		}
	}
	waitKey(0);
	return 0;
}