#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/photo.hpp>
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
	inRange(downscaled, Scalar(0, 0, 0), Scalar(180, 10, 255), downscaled);

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
		/*static Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * 6 + 1, 2 * 6 + 1),
			Point(6, 6));
		erode(mask, mask, element);*/
		if(show_imgs)
			imshow("roundpen mask", mask);
	}
	output_hsv = Mat();
	input_hsv.copyTo(output_hsv, mask);
}

int marker_removing(Mat& img, Mat& out, vector<rp::Marker>& markers, int frameNr) {
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8U);

	for (auto& marker : markers) {
		if (marker.is_trackable(frameNr)) {
			if (marker.is_tracked_for_frame(frameNr)) {
				circle(mask, marker.get_last_position(), 13, Scalar(255), FILLED);
			}
			else {
				auto pos = marker.get_next_position(frameNr);
				auto diff = pos - marker.get_last_position();
				circle(mask, pos, max(13, (int)sqrt(pow(diff.x, 2) + pow(diff.y, 2))), Scalar(255), FILLED);
			}
		}
	}

	inpaint(img, mask, out, 12, INPAINT_TELEA);
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
	cv::Vec3b background;
	vector<rp::Marker> markers;

	{
		char nameBuffer[40];
		char h[4], s[4], v[4];

		while (reader.getline(nameBuffer, 40, ';')) {
			reader.getline(h, 4, ';');
			reader.getline(s, 4, ';');
			reader.getline(v, 4);
			if (strcmp(nameBuffer, "Background") == 0) {
				background = cv::Vec3b(atoi(h), atoi(s), atoi(v));
			}
			else {
				markers.push_back(rp::Marker(cv::String(nameBuffer), cv::Vec3b(atoi(h), atoi(s), atoi(v))));
			}
		}
		reader.close();
	}

	VideoCapture cap;
	Mat frame;
	Mat frame_hsv;
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
		resize(frame, frame_points, Size(800, (int)(800 * ratio)));
		imshow("Image", frame_points);
		if(waitKey(20) == ' ') break;
	}
	namedWindow("Cut", WINDOW_NORMAL);
	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Output", WINDOW_NORMAL);
	resizeWindow("Cut", 400, (int)(400 * ratio));
	resizeWindow("Input", 800, (int)(800 * ratio));
	resizeWindow("Output", 800, (int)(800 * ratio));

	VideoWriter outWriter("output_video.avi", VideoWriter::fourcc('W', 'M', 'V', '2'), 30, frame.size());
	ofstream outMarker;
	outMarker.open("output_markers.csv");

	for (int i = 0; i < markers.size(); i++) {
		outMarker << markers[i].get_name() << ";";
	}
	outMarker << endl;
#pragma omp parallel
#pragma omp master
	while (cap.read(frame)) {
		//frame.copyTo(frame_points);
		resize(frame, frame_points, Size(800, (int)(800 * ratio)));
		imshow("Input", frame_points);

		cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
		// Find RoundPen
		resize(frame_hsv, downscaled, Size(100, (int)(100 * ratio)));

		cut_roundPen(frame_hsv, frame_only_roundpen, mask, downscaled, contours, hull);

		resize(frame_only_roundpen, downscaled, Size(100, (int)(100 * ratio)));
		imshow("Cut", downscaled);
		
		if (init) {
			// Calibrate optimal marker color filter ranges.
			// This will take much time.
#pragma omp parallel for
			for (int i = 0; i < markers.size(); i++) {
				markers[i].calibrate_marker_range(frame_only_roundpen, frameNr);
			}
			init = false;
		}
		else {

			//GaussianBlur(frame_only_roundpen, frame_only_roundpen, Size(5, 5), 0);
#pragma omp parallel for
			for (int i = 0; i < markers.size(); i++) {
				Point2d pos;
				int validity = markers[i].find_position(frame_only_roundpen, frameNr, &pos);
				if (validity == 0) {
					markers[i].set_current_position(frameNr, &pos);
				}
				else {
					markers[i].calibrate_marker_range(frame_only_roundpen, frameNr);
				}
			}
		}

		for (int i = 0; i < markers.size(); i++) {
			ostringstream markerText;
			auto data = markers[i].get_marker_color();
			markerText << markers[i].get_name();
			if (markers[i].is_trackable(frameNr)) {
				auto pos = markers[i].get_next_position(frameNr);
				markerText << ": " << (int) pos.x << ", " << (int) pos.y;
				drawMarker(frame_points, pos / (((double)frame.cols) / 800), cv::Scalar(0, 0, 128 + ((markers[i].is_tracked_for_frame(frameNr)) ? 127 : 0)), cv::MARKER_TILTED_CROSS, 5, 2);
			}
			putText(frame_points, markerText.str(), cv::Point(10, i * 10 + 10), FONT_HERSHEY_PLAIN, 0.5, rp::ScalarHSV2BGR(data[0], data[1], data[2]), 1);
		}

		imshow("Image", frame_points);
		frameNr++;

		// Copy to a dedicated memory to put it into a new task.
		/*Mat frame_output;
		vector<rp::Marker> frame_output_markers;

		frame.copyTo(frame_output);
		for (int i = 0; i < markers.size(); i++) {
			frame_output_markers.push_back(markers[i]);
		}*/
//#pragma omp task untied
		{
				marker_removing(frame, frame, markers, frameNr);

				resize(frame, frame_points, Size(800, (int)(800 * ratio)));
				imshow("Output", frame_points);

				outWriter.write(frame);
				for (int i = 0; i < markers.size(); i++) {
					if (markers[i].is_trackable(frameNr)) {
						if (markers[i].is_tracked_for_frame(frameNr)) {
						}
						else {
							auto pos = markers[i].get_next_position(frameNr);
							outMarker << pos.x << ", " << pos.y << "*";
						}
					}
					outMarker << ";";
				}
				outMarker << endl;
		}
		if(waitKey(1) == 27) break;
	}
	outWriter.release();
	outMarker.close();
	destroyWindow("Input");
	destroyWindow("Cut");
	destroyWindow("Output");
	putText(frame_points, "Tracking done.", Point(frame_points.cols / 2 - 100, frame_points.rows / 2), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
	imshow("Image", frame_points);
	waitKey(0);
	return 0;
}