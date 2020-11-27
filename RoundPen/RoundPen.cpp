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

void cut_roundPen(Mat& input_hsv, Mat& output_hsv, Mat& mask, Mat& downscaled, vector<vector<Point>>& contours, vector<vector<Point>>& hull, bool show_imgs = false) {
	int contourID = -1;
	double area, arclength, circularity;
	double prevCircularity = 0;

	//Point2f roundPenCenter;
	//float roundPenRadius;

	//GaussianBlur(frame, frame_gaussian, Size(45, 45), 0);
	inRange(downscaled, Scalar(5, 0, 0), Scalar(30, 40, 255), downscaled);

	findContours(downscaled, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if (area > 1000) {
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
		if (show_imgs) {
			drawContours(downscaled, contours, -1, Scalar(128), -1);
			drawContours(downscaled, contours, contourID, Scalar(255), -1);
			imshow("tmp", downscaled);
		}
		convexHull(contours[contourID], hull[0]);
		//drawContours(mask, hull, 0, Scalar(255), -1);
		fillConvexPoly(mask, Mat(hull[0]) * (((double)input_hsv.cols) / 100), Scalar(255));
		static Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * 6 + 1, 2 * 6 + 1),
			Point(6, 6));
		erode(mask, mask, element);
		if(show_imgs)
			imshow("roundpen mask", mask);
	}
	input_hsv.copyTo(output_hsv, mask);
}

int find_center_of_marker(rp::Marker& marker, int frameNr, Mat& frame, Mat& frame_thresh) {
	bool using_previous_pos = marker.is_trackable(frameNr);

	if (using_previous_pos) {

	}
	return 0;
}

int main(int argn, char** argv)
{
	String path = "./RoundPen.mov";
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

	char nameBuffer[40];
	char h[3], s[3], v[3];
	cv::Vec3b background;

	vector<rp::Marker> markers;
	while (reader.getline(nameBuffer, 40, ';')) {
		reader.getline(h, 40, ';');
		reader.getline(s, 40, ';');
		reader.getline(v, 40);
		if (strcmp(nameBuffer, "Background") == 0) {
			background = cv::Vec3b(atoi(h), atoi(s), atoi(v));
		}
		else {
			markers.push_back(rp::Marker(cv::String(nameBuffer), cv::Vec3b(atoi(h), atoi(s), atoi(v))));
		}
	}
	reader.close();

	VideoCapture cap;
	Mat frame;
	Mat mask;
	Mat frame_points;
	Mat downscaled;
	Mat frame_only_roundpen;
	vector<vector<Point>> contours;
	vector<vector<Point>> hull(1);

	cap.open(path);
	if (!cap.isOpened()) {
		cerr << "Error opening video" << endl;
		cerr << "Call the command with a valid video file as first parameter" << endl;
		return -1;
	}

	namedWindow("Image", WINDOW_NORMAL);

	double ratio = 0;
	bool init = true;
	uint8_t frameNr = 0;
	while (cap.read(frame)) {
		if (ratio == 0) {
			ratio = ((double) frame.rows) / frame.cols;
			resizeWindow("Image", 800, (int) (800 * ratio));
		}
		imshow("Image", frame);
		if(waitKey(20) == ' ') break;
	}
	namedWindow("Cut", WINDOW_NORMAL);
	namedWindow("Input", WINDOW_NORMAL);
	resizeWindow("Cut", 400, (int)(400 * ratio));
	resizeWindow("Input", 400, (int)(400 * ratio));

	while (cap.read(frame)) {
		//frame.copyTo(frame_points);
		resize(frame, frame_points, Size(800, (int)(800 * ratio)));
		cvtColor(frame, frame, COLOR_BGR2HSV);
		// Find RoundPen
		resize(frame, downscaled, Size(100, (int)(100 * ratio)));

		imshow("Input", downscaled);
		cut_roundPen(frame, frame_only_roundpen, mask, downscaled, contours, hull);

		resize(frame_only_roundpen, downscaled, Size(100, (int)(100 * ratio)));
		imshow("Cut", downscaled);
		
		if (init) {
#pragma omp parallel for
			for (int i = 0; i < markers.size(); i++) {
				markers[i].calibrate_marker_range(frame_only_roundpen);
			}
			init = false;
		}
		
		//GaussianBlur(frame_only_roundpen, frame_only_roundpen, Size(5, 5), 0);
#pragma omp parallel for
		for (int i = 0; i < markers.size(); i++) {
			Mat frame_marker_area;
			Mat frame_thresh;

			Moments mu;
			Point2d* prevCenter;
			Point2d pCenterCache;
			vector<vector<Point>> cont;

			int xLow = 0, yLow = 0;
			prevCenter = nullptr;
			if (markers[i].is_trackable(frameNr)) {
				pCenterCache = markers[i].get_next_position(frameNr);
				auto prevKnownPos = markers[i].get_last_position();

				auto dist = prevKnownPos - pCenterCache;

				prevCenter = &pCenterCache;
				xLow = max(0.0, min(pCenterCache.x, (double)frame_only_roundpen.cols) - 40), yLow = max(0.0, min(pCenterCache.y, (double)frame_only_roundpen.rows) - 40);
				auto maxX = min(frame_only_roundpen.cols - xLow - 1, 80);
				auto maxY = min(frame_only_roundpen.rows - yLow - 1, 80);

				frame_marker_area = frame_only_roundpen(Rect(xLow, yLow, maxX, maxY));
			}
			else {
				frame_marker_area = frame_only_roundpen;
			}
			inRange(frame_marker_area, markers[i].get_marker_color_range_low(), markers[i].get_marker_color_range_high(), frame_thresh);
			Mat element = getStructuringElement(MORPH_ELLIPSE,
				Size(2 * 2 + 1, 2 * 2 + 1),
				Point(2, 2));
			dilate(frame_thresh, frame_thresh, element);
			findContours(frame_thresh, cont, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

			if (cont.size() == 0) {
				//markers[i].calibrate_marker_range(frame_only_roundpen);
			}
			else if (cont.size() >= 1) {
				int area;
				double contourX, contourY;
				double contourError;
				double error = INFINITY;
				for (int i = 0; i < cont.size(); i++) {
					area = contourArea(cont[i]);
					if (area < 200) {
						mu = moments(cont[i]);
						contourX = max(0.0, min(mu.m10 / (mu.m00 + 1e-5) + xLow, (double)frame_only_roundpen.cols));
						contourY = max(0.0, min(mu.m01 / (mu.m00 + 1e-5) + yLow, (double)frame_only_roundpen.rows));
						contourError = pow(contourX - (xLow + 40), 2) + pow(contourY - (yLow + 40), 2);
						if (prevCenter == nullptr || contourError < error) {
							error = contourError;
							prevCenter = &pCenterCache;
							prevCenter->x = contourX;
							prevCenter->y = contourY;
						}
					}
				}
				if (prevCenter != nullptr) {
					//drawMarker(frame_points, (*prevCenter) / (((double)frame.cols) / 800), cv::Scalar(0, 0, 128), cv::MARKER_TILTED_CROSS, 5, 2);
				}
				markers[i].set_current_position(frameNr, prevCenter);
			}
		}

		for (int i = 0; i < markers.size(); i++) {
			ostringstream markerText;
			auto data = markers[i].get_marker_color();
			markerText << markers[i].get_name();
			if (markers[i].is_trackable(frameNr)) {
				auto pos = markers[i].get_next_position(frameNr);
				markerText << ": " << (int) pos.x << ", " << (int) pos.y;
				drawMarker(frame_points, pos / (((double)frame.cols) / 800), cv::Scalar(0, 0, 128), cv::MARKER_TILTED_CROSS, 5, 2);
			}
			putText(frame_points, markerText.str(), cv::Point(10, i * 10 + 10), FONT_HERSHEY_PLAIN, 0.5, rp::ScalarHSV2BGR(data[0], data[1], data[2]), 1);
		}

		imshow("Image", frame_points);
		frameNr++;
		waitKey(1);
	}
	waitKey(0);
	return 0;
}