#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "Marker.h"

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
	String path = "./RoundPen.mp4";
	String markersFile = "./markers.csv";
	for (int i = 2; i < argn; i += 2) {
		if (argv[i-1] == "--video") {
			path = argv[i];
		}
		else if (argv[i-1] == "--markers") {
			markersFile = argv[i];
		}
	}

	ifstream reader;
	reader.open(markersFile);
	if (!reader.is_open()) return 1;

	char markerName[40];
	char h[3], s[3], v[3];

	vector<rp::Marker> markers;
	while (reader.getline(markerName, 40, ';')) {
		reader.getline(h, 40, ';');
		reader.getline(s, 40, ';');
		reader.getline(v, 40);
		markers.push_back(rp::Marker(cv::String(markerName), cv::Vec3b(atoi(h), atoi(s), atoi(v))));
	}
	reader.close();

	VideoCapture cap;
	Mat frame;
	Mat mask;
	Mat downscaled;
	Mat frame_only_roundpen;
	Mat frame_thresh;
	Mat frame_marker_area;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> hull(1);

	cap.open(path);
	if (!cap.isOpened()) {
		cerr << "Error opening video" << endl;
		cerr << "Call the command with a valid video file as first parameter" << endl;
		return -1;
	}

	bool init = true;
	uint8_t frameNr = 0;

	while (cap.read(frame)) {
		imshow("img", frame);
		cvtColor(frame, frame, COLOR_BGR2HSV);
		// Find RoundPen
		cut_roundPen(frame, frame_only_roundpen, mask, downscaled, contours, hull, hierarchy);

		imshow("img_cut", frame_only_roundpen);

		if (init) {
			for (auto marker : markers) {
				marker.calibrate_marker_range(frame_only_roundpen);
			}
			init = false;
		}
		for (auto marker : markers) {
			int xLow = 0, yLow = 0;
			if (marker.is_trackable()) {
				auto nextPos = marker.get_next_position(frameNr);
				xLow = nextPos.x - 40, yLow = nextPos.y - 40;
				frame_marker_area = frame_only_roundpen(Rect(xLow, yLow, 80, 80));
			}
			else {
				frame_marker_area = frame_only_roundpen;
			}
			inRange(frame_marker_area, marker.get_marker_color_range_low(), marker.get_marker_color_range_high(), frame_thresh);
			findContours(frame_thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

			Moments m;
			double centerX, centerY;
			if (contours.size == 0) {
				marker.calibrate_marker_range(frame_only_roundpen);
			} else if(contours.size > 1 && xLow != 0 && yLow != 0) {
				int area;
				double contourX, contourY;
				double contourError;
				double error = INFINITY;
				for (int i = 0; i < contours.size; i++) {
					area = contourArea(contours[i]);
					if (area < 50) {
						m = moments(contours[i]);
						contourX = m.m10 / m.m00;
						contourY = m.m01 / m.m00;
						contourError = pow(contourX - (xLow + 40), 2) + pow(contourY - (yLow + 40), 2);
						if (contourError < error) {
							error = contourError;
							centerX = contourX;
							centerY = contourY;
						}
					}
				}
				marker.set_current_position(frameNr, &Point2d(centerX, centerY));
			}
			else if (contours.size == 1) {
				m = moments(contours[0]);
				centerX = m.m10 / m.m00;
				centerY = m.m01 / m.m00;
				marker.set_current_position(frameNr, &Point2d(centerX, centerY));
			}
		}
		frameNr++;
	}
	waitKey(0);
	return 0;
}