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

#ifdef _DEBUG
#define STOPABLE // CAN TRACKING BE STOPPED? Will result in an additional key check.
#define INFO_IMG // Show markers while tracking
#define DEBUG_IMG // Show intermediate results
//#define DEBUG_CONSOLE // Show info on console
#elif NDEBUG
#define INFO_CONSOLE // Show progress.
#endif


void cut_roundPen(Mat& input_hsv, Mat& output_hsv, Mat& mask, Mat& downscaled, vector<vector<Point>>& contours, vector<vector<Point>>& hull, bool show_imgs = false) {
	int contourID = -1;
	double area, arclength, circularity;
	double prevCircularity = 0;

	inRange(downscaled, Scalar(0, 0, 0), Scalar(180, 14, 255), downscaled);
	findContours(downscaled, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		// TODO FIXED AREA SIZE FOR ROUNDPEN.
		if (area > 500) {
			arclength = arcLength(contours[i], true);
			circularity = 4 * CV_PI * area / (arclength * arclength);
			if (circularity > prevCircularity) {
				prevCircularity = circularity;
				contourID = i;
			}
		}
	}

	// RESET MASK.
	mask = Mat::zeros(input_hsv.size(), CV_8U);
	if (contourID >= 0) {
#ifdef DEBUG_IMG
		namedWindow("Roundpen contour", cv::WINDOW_NORMAL);
		resizeWindow("Roundpen contour", 600, (int)(600 * downscaled.rows / downscaled.cols));
		drawContours(downscaled, contours, -1, Scalar(128), -1);
		drawContours(downscaled, contours, contourID, Scalar(255), -1);
		imshow("Roundpen contour", downscaled);
#endif
		
		convexHull(contours[contourID], hull[0]);
		//drawContours(mask, hull, 0, Scalar(255), -1);
		fillConvexPoly(mask, Mat(hull[0]) * (((double)input_hsv.cols) / 100), Scalar(255));
#ifdef DEBUG_IMG
		cv::namedWindow("Roundpen mask", cv::WINDOW_NORMAL);
		cv::resizeWindow("Roundpen mask", 600, (int)(600 * mask.rows / mask.cols));
		imshow("Roundpen mask", mask);
#endif
	}
	// RESET OUTPUT_HSV
	output_hsv = Mat();
	// FILL MASK AREAS IN OUTPUT_HSV
	input_hsv.copyTo(output_hsv, mask);
}

int make_mask(Mat& img, Mat& mask, vector<rp::Marker>& markers, int frameNr) {
	for (auto& marker : markers) {
		if (marker.is_trackable(frameNr)) {
			if (marker.is_tracked_for_frame(frameNr)) {
				circle(mask, marker.get_last_position(), (int)(marker.get_marker_radius() + 8), Scalar(255, 255, 255), FILLED);
			}
			else {
				auto pos = marker.get_next_position(frameNr);
				// Only draw if predicted pos is in frame
				if (pos.x < img.cols && pos.y < img.rows && pos.x >= 0 && pos.y >= 0) {
					//auto diff = pos - marker.get_last_position();
					circle(mask, pos, (int)(marker.get_marker_radius() + 8/* + (int)std::max(diff.x, diff.y)*/), Scalar(255, 255, 255), FILLED);
				}
			}
		}
	}
	return 0;
}

/*int marker_removing(Mat& img, Mat& out, vector<rp::Marker>& markers, int frameNr) {
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
}*/

int main(int argn, char** argv)
{
	// READING CONSOLE INPUT.
	String path = "./RoundPen.mov";
	String outputpath = "./";
	String markersFile = "./markers.csv";
	char codec[4] = { 'D', 'I', 'V', 'X'};
	int offset = -1, finish = -1;
	if (argn == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
		cout << "Starts the Roundpen Marker Tracking" << endl
			<< '\t' << "--video VIDEOFILE" << endl
			<< "\t\t" << "path to the video file to track [*.mov/*.avi/*.mp4/...] (default ./RoundPen.mov)" << endl
			<< '\t' << "--markers MARKERFILE" << endl 
			<< "\t\t" << "path to the marker info file [*.csv] default (default: ./markers.csv)" << endl
			<< '\t' << "--output OUTPUTPATH"<< endl 
			<< "\t\t" << "path to the output location [*] default (default: ./)" << endl
			<< '\t' << "--codec CODEC" << endl 
			<< "\t\t" << "defines the video codec for the output [MJPG/DIVX/...] (default: DIVX)" << endl
			<< '\t' << "--offset OFFSET" << endl
			<< "\t\t" << "defines the frames offset at the beginning [0-...] (default: -1)" << endl
			<< '\t' << "--finish FINISHFRAME" << endl
			<< "\t\t" << "defines at which frame it should end [0-...] (default: -1)" << endl;
		return 0;
	}
	for (int i = 2; i < argn; i += 2) {
		if (strcmp(argv[i-1], "--video") == 0) {
			path = argv[i];
		}
		else if (strcmp(argv[i-1], "--markers") == 0) {
			markersFile = argv[i];
		}
		else if (strcmp(argv[i - 1], "--codec") == 0) {
			codec[0] = argv[i][0];
			codec[1] = argv[i][1];
			codec[2] = argv[i][2];
			codec[3] = argv[i][3];
		}
		else if (strcmp(argv[i - 1], "--output") == 0) {
			outputpath = argv[i];
		}
		else if (strcmp(argv[i - 1], "--offset") == 0) {
			offset = atoi(argv[i]);
		}
		else if (strcmp(argv[i - 1], "--finish") == 0) {
			finish = atoi(argv[i]);
		}
	}

	// READING MARKER FILE.
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

	// OPEN VIDEO.
	VideoCapture cap;
	Mat frame;
	Mat frame_hsv;
	Mat mask;
	Mat frame_points;
	Mat downscaled;
	Mat frame_only_roundpen;
	vector<vector<Point>> contours;
	vector<vector<Point>> hull(1);
	double ratio = 0;
	bool init = true;
	uint64_t frameNr = 0;

	cap.open(path);
	if (!cap.isOpened()) {
		cerr << "Error opening video" << endl;
		cerr << "Call the command with a valid video file as first parameter" << endl;
		return -1;
	}
	ratio = cap.get(CAP_PROP_FRAME_HEIGHT) / cap.get(CAP_PROP_FRAME_WIDTH);

	// PLAYING VIDEO UNTIL SPACE IS PRESSED.
	if (offset == -1) {
		offset = 0;
		namedWindow("Image", WINDOW_NORMAL);
		resizeWindow("Image", 800, (int)(800 * ratio));

		bool startTracking = false;
		while (!startTracking && cap.read(frame)) {
			resize(frame, frame_points, Size(800, (int)(800 * ratio)));
			imshow("Image", frame_points);
			offset++;

			switch (waitKey(20)) {
			case ' ': startTracking = true; break;
			case 27: return 0;
			}
		}
#ifndef INFO_IMG
		// Destroy window if it isn't used later on.
		destroyWindow("Image");
		frame_points.release();
#endif
	}
	else {
#ifdef INFO_IMG
		namedWindow("Image", WINDOW_NORMAL);
		resizeWindow("Image", 800, (int)(800 * ratio));
#endif
		cap.set(CAP_PROP_POS_FRAMES, offset);
	}

	// PREPARE DIFFERENT LOG LEVELS.
#ifdef DEBUG_IMG
	namedWindow("Cut", WINDOW_NORMAL);
	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Output", WINDOW_NORMAL);
	resizeWindow("Cut", 400, (int)(400 * ratio));
	resizeWindow("Input", 800, (int)(800 * ratio));
	resizeWindow("Output", 800, (int)(800 * ratio));
#endif
#ifdef INFO_CONSOLE
	cout << "Starting with a video offset of " << offset << endl;
	uint64_t vidlength = (uint64_t)cap.get(CAP_PROP_FRAME_COUNT) - offset;
	if (finish != -1) {
		vidlength = finish - offset;
		cout << "Video ending at " << finish << endl;
		finish -= offset;
	}
#endif

	// OPENING OUTPUT STREAMS.
	Size vidSize = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), (int) cap.get(CAP_PROP_FRAME_HEIGHT));
	VideoWriter outWriter(outputpath + "output_video.avi", VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]), 30, vidSize);
	VideoWriter maskWriter(outputpath + "mask_video.avi", VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]), 30, vidSize);
	ofstream marWriter;
	marWriter.open("output_markers.csv");

	// PREPARING MARKER OUTPUT FILES.
	for (int i = 0; i < markers.size(); i++) {
		marWriter << markers[i].get_name() << ";";
	}
	marWriter << endl;


	// START TRACKING MARKERS.
#ifndef DEBUG_IMG
#pragma omp parallel
#pragma omp master
#endif
	while (frameNr != finish && cap.read(frame)) {
#if defined(INFO_IMG) || defined(DEBUG_IMG)
		resize(frame, frame_points, Size(800, (int)(800 * ratio)));
#endif
#ifdef DEBUG_IMG
		imshow("Input", frame_points);
#endif

		cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
		// Find RoundPen
		resize(frame_hsv, downscaled, Size(100, (int)(100 * ratio)));

		cut_roundPen(frame_hsv, frame_only_roundpen, mask, downscaled, contours, hull);
#ifdef DEBUG_IMG
		imshow("Cut", frame_only_roundpen);
#endif

		resize(frame_only_roundpen, downscaled, Size(100, (int)(100 * ratio)));
		
		if (init) {
			// Calibrate optimal marker color filter ranges.
			// This will take much time.
#ifndef DEBUG_IMG
#pragma omp parallel for
#endif
			for (int i = 0; i < markers.size(); i++) {
				markers[i].calibrate_marker_range(frame_only_roundpen, (uint8_t) frameNr);
			}
			init = false;
		}
		else {
#ifndef DEBUG_IMG
#pragma omp parallel for
#endif
			for (int i = 0; i < markers.size(); i++) {
				Point2d pos;
				int validity = markers[i].find_position(frame_only_roundpen, (uint8_t) frameNr, &pos);
				if (validity == 0) {
					markers[i].set_current_position((uint8_t) frameNr, &pos);
				}
				else {
					markers[i].calibrate_marker_range(frame_only_roundpen, (uint8_t) frameNr);
				}
			}
		}

#ifdef INFO_IMG
		for (int i = 0; i < markers.size(); i++) {
			ostringstream markerText;
			auto data = markers[i].get_marker_color();
			markerText << markers[i].get_name();
			if (markers[i].is_trackable((uint8_t) frameNr)) {
				auto pos = markers[i].get_next_position((uint8_t) frameNr);
				markerText << ": " << (int) pos.x << ", " << (int) pos.y;
				drawMarker(frame_points, pos / (((double)frame.cols) / 800), cv::Scalar(0, 0, 128 + ((markers[i].is_tracked_for_frame((uint8_t) frameNr)) ? 127 : 0)), cv::MARKER_TILTED_CROSS, 5, 2);
			}
			putText(frame_points, markerText.str(), cv::Point(10, i * 10 + 10), FONT_HERSHEY_PLAIN, 0.5, rp::ScalarHSV2BGR(data[0], data[1], data[2]), 1);
		}

		imshow("Image", frame_points);
#endif

		// Copy to a dedicated memory to put it into a new task.
		/*Mat frame_output;
		vector<rp::Marker> frame_output_markers;

		frame.copyTo(frame_output);
		for (int i = 0; i < markers.size(); i++) {
			frame_output_markers.push_back(markers[i]);
		}*/
//#pragma omp task untied
		{
			//Deactivate inpainting so we can use it for other approaches.
			/* 
			marker_removing(frame, frame, markers, frameNr);

			resize(frame, frame_points, Size(800, (int)(800 * ratio)));
			imshow("Output", frame_points);
			*/
			mask = Mat::zeros(frame.size(), CV_8UC3);
			make_mask(frame, mask, markers, (uint8_t)frameNr);
#ifdef DEBUG_IMG
			imshow("Output", mask);
#endif

			outWriter.write(frame);
			maskWriter.write(mask);
			for (int i = 0; i < markers.size(); i++) {
				if (markers[i].is_trackable((uint8_t)frameNr)) {
					if (markers[i].is_tracked_for_frame((uint8_t)frameNr)) {
						auto pos = markers[i].get_last_position();
						marWriter << pos.x << ", " << pos.y;
					}
					else {
						auto pos = markers[i].get_next_position((uint8_t)frameNr);
						marWriter << "(" << pos.x << ", " << pos.y << ")";
					}
				}
				marWriter << ";";
			}
			marWriter << endl;
		}
#ifdef INFO_CONSOLE
		cout << frameNr << "/" << vidlength << '\r';
#endif

		frameNr++;
#ifdef STOPABLE
		if(waitKey(1) == 27) break;
#endif
	}
#ifdef INFO_CONSOLE
	// Opencv can be inaccurate about the max number of frames.
	// This will show the correct result and places the cursor correctly.
	cout << vidlength << "/" << vidlength << endl;
#endif
	outWriter.release();
	maskWriter.release();
	marWriter.close();
#ifdef DEBUG_IMG
	destroyWindow("Input");
	destroyWindow("Cut");
	destroyWindow("Output");
#endif
#ifdef INFO_IMG
	putText(frame_points, "Tracking done.", Point(frame_points.cols / 2 - 100, frame_points.rows / 2), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
	imshow("Image", frame_points);
	waitKey(0);
	destroyWindow("Image");
#endif
	return 0;
}