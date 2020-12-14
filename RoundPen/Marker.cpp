#include <iostream>
#include <stdio.h>

#include "Marker.h"

#ifdef _DEBUG
//#define DEBUG_IMG // Show intermediate results
//#define DEBUG_CONSOLE // Show info on console
#elif NDEBUG
#define INFO_CONSOLE // Show progress.
#endif

#ifdef DEBUG_IMG
#include <opencv2/highgui/highgui.hpp>
#endif

namespace rp {
	const int MINIMAL_HSV_RANGE = 3;
	const int START_H_RANGE_LOW = 10;
	const int START_H_RANGE_HIGH = 10;
	const int START_S_RANGE_LOW = 10;
	const int START_S_RANGE_HIGH = 40;
	const int START_V_RANGE_LOW = 40;
	const int START_V_RANGE_HIGH = 40;

	void optimize_search_area_space(cv::Mat area, cv::Point offset, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Mat>& output, std::vector<cv::Point>& outOffset) {
		std::vector<cv::Rect> tempRects;
		for (int i = 0; i < contours.size(); i++) {
			cv::Rect r = cv::boundingRect(contours[i]);
			if (r.area() < 2) continue; // Area probably invalid as it is too small.
			bool addedToExistingArea = false;
			for (int j = 0; j < tempRects.size(); j++) {
				if ((r & tempRects[j]).area() > 0) {
					int xEnd = std::max(tempRects[j].x + tempRects[j].width, r.x + r.width);
					int yEnd = std::max(tempRects[j].y + tempRects[j].height, r.y + r.height);
					tempRects[j].x = std::min(r.x, tempRects[j].x);
					tempRects[j].y = std::min(r.y, tempRects[j].y);

					tempRects[j].width = xEnd - tempRects[j].x;
					tempRects[j].height = yEnd - tempRects[j].y;
					addedToExistingArea = true;
					break;
				}
			}
			if (!addedToExistingArea) {
				// We increase the area a little bit to ignore minimal errors in detection.
				auto xPos = std::max(0, r.x - 5);
				auto yPos = std::max(0, r.y - 5);
				auto wid = std::min(r.width + 10, area.cols - xPos);
				auto hei = std::min(r.height + 10, area.rows - yPos);
				tempRects.push_back(cv::Rect(xPos, yPos, wid, hei));
			}
		}
		for (int i = 0; i < tempRects.size(); i++) {
			output.push_back(area(tempRects[i]));
			outOffset.push_back(cv::Point(tempRects[i].x + offset.x, tempRects[i].y + offset.y));
		}
	}

	// If marker color is unique, threshold will only use H channel.
	// If marker saturation is unique, threshold will use HS channel.
	// If marker brightness is unique, threshold will use HSV channel.
	void Marker::calibrate_marker_range(cv::Mat& img, uint8_t frameNr) {
		// We work on tmp.
		cv::Mat tmp;
		cv::Mat rangeOut;
		std::vector<std::vector<cv::Point>> contours;
		mCLow = cv::Vec3b(FIX_SUBR_H_RANGE(mHsvColor[0] - START_H_RANGE_LOW), FIX_SUBR_SV_RANGE(mHsvColor[1], START_S_RANGE_LOW), FIX_SUBR_SV_RANGE(mHsvColor[2], START_V_RANGE_LOW));
		mCHigh = cv::Vec3b(FIX_ADD_H_RANGE(mHsvColor[0] + START_H_RANGE_HIGH), FIX_ADD_SV_RANGE(mHsvColor[1], START_S_RANGE_HIGH), FIX_ADD_SV_RANGE(mHsvColor[2], START_V_RANGE_HIGH));
		img.copyTo(tmp);

		if (mCLow[1] < 8) {
			mCLow[1] = 8;
		}

		cv::inRange(tmp, mCLow, mCHigh, rangeOut);

		cv::findContours(rangeOut, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
#ifdef DEBUG_IMG
		cv::Mat out;
		img.copyTo(out);
		cv::namedWindow("Color Search Area", cv::WINDOW_NORMAL);
		cv::resizeWindow("Color Search Area", 600, (int)(600 * out.rows / out.cols));
		cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), -1);
#endif

		std::vector<cv::Mat> contourAreas;
		std::vector<cv::Point> contourOffsets;

		if (contours.size() < 200) {
			optimize_search_area_space(tmp, cv::Point(0, 0), contours, contourAreas, contourOffsets);
		}
		else {
			contourAreas.push_back(tmp);
		}

		std::vector<std::vector<cv::Point>> contoursInsideContour;
		cv::Mat contourAreaThresh;

		int i = 0;
		// Optimize H until good enough or impossible.
		while (mCLow[0] != mHsvColor[0] - MINIMAL_HSV_RANGE && mCHigh[0] != mHsvColor[0] + MINIMAL_HSV_RANGE && contourAreas.size() > 1) {
			mCLow[0] = FIX_ADD_H_RANGE(mCLow[0] + 1);
			mCHigh[0] = FIX_SUBR_H_RANGE(mCHigh[0] - 1);

			for (int j = 0; j < contourAreas.size(); j++) {
				cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

				cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
#ifdef DEBUG_IMG
				cv::drawContours(out, contoursInsideContour, -1, cv::Scalar(0, 0, 255), 1, 8, cv::noArray(), 2147483647, cv::Point(contourOffsets[j]));
				cv::imshow("Color Search Area", out);
#endif
				if (contoursInsideContour.size() < 200) {
					optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
					contourAreas.erase(contourAreas.begin() + j);
					contourOffsets.erase(contourOffsets.begin() + j);
				}
			}

			i++;
		}

		// If no good optimization, optimize L until good enough or impossible.
		if (mCLow[0] == mHsvColor[0] - MINIMAL_HSV_RANGE || mCHigh[0] == mHsvColor[0] + MINIMAL_HSV_RANGE) {
			int i = 0;
			// Optimize H until good enough or impossible.
			while (mCLow[1] != mHsvColor[1] - MINIMAL_HSV_RANGE && mCHigh[1] != mHsvColor[1] + MINIMAL_HSV_RANGE && contourAreas.size() > 1) {
				mCLow[1] = mCLow[1] + 1;
				mCHigh[1] = mCHigh[1] - 1;

				for (int j = 0; j < contourAreas.size(); j++) {
					cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

					cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
#ifdef DEBUG_IMG
					cv::drawContours(out, contoursInsideContour, -1, cv::Scalar(0, 0, 255), 1, 8, cv::noArray(), 2147483647, cv::Point(contourOffsets[j]));
					cv::imshow("Color Search Area", out);
#endif
					if (contoursInsideContour.size() < 200) {
						optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
						contourAreas.erase(contourAreas.begin() + j);
						contourOffsets.erase(contourOffsets.begin() + j);
					}
				}

				i++;
			}

			// If no good optimization, optimize S until good enough.
			if (mCLow[1] == mHsvColor[1] - MINIMAL_HSV_RANGE || mCHigh[1] == mHsvColor[1] + MINIMAL_HSV_RANGE) {
				int i = 0;
				// Optimize H until good enough or impossible.
				while (mCLow[2] != mHsvColor[2] - MINIMAL_HSV_RANGE && mCHigh[2] != mHsvColor[2] + MINIMAL_HSV_RANGE && contourAreas.size() > 1) {
					mCLow[2] = mCLow[2] + 1;
					mCHigh[2] = mCHigh[2] - 1;

					for (int j = 0; j < contourAreas.size(); j++) {
						cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

						cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
#ifdef DEBUG_IMG
						cv::drawContours(out, contoursInsideContour, -1, cv::Scalar(0, 0, 255), 1, 8, cv::noArray(), 2147483647, cv::Point(contourOffsets[j]));
						cv::imshow("Color Search Area", out);
#endif
						if (contoursInsideContour.size() < 200) {
							optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
							contourAreas.erase(contourAreas.begin() + j);
							contourOffsets.erase(contourOffsets.begin() + j);
						}
					}

					i++;
				}
			}
		}

		if (contourAreas.size() == 1) {
			set_current_position(frameNr, &cv::Point2d(contourOffsets[0].x + contourAreas[0].cols / 2, contourOffsets[0].y + contourAreas[0].rows / 2));
			mMarkerRadius = (double) std::max(1, std::max(contourAreas[0].cols / 2, contourAreas[0].rows / 2));
		}
		else {
			// This updates mTracked. If point is lost, this will keep it lost.
			is_trackable(frameNr);
		}
#ifdef DEBUG_IMG
		cv::imshow("Color Search Area", out);
		cv::waitKey(0);
#endif
		// Range optimized on first frame.
		// Program itself has better detection as too large areas will just be ignored.
	}

	void Marker::append_neighbor(MarkerEdge* edge) {
		mNeighbors.push_back(edge);
	}

	cv::Vec3b Marker::get_marker_color() {
		return mHsvColor;
	}

	cv::Vec3b Marker::get_marker_color_range_low() {
		return mCLow;
	}

	cv::Vec3b Marker::get_marker_color_range_high() {
		return mCHigh;
	}

	cv::Point2d Marker::get_last_position() {
		return mLastPosition;
	}

	bool Marker::is_trackable(int currentFrame) {
		mTracked = mTracked && ((currentFrame - mLastFrame) < mFramesUntilLost);
		if (!mTracked) {
			mLastAcceleration = cv::Point2d(0, 0);
			mLastVelocity = cv::Point2d(0, 0);
		}
		return mTracked;
	}

	cv::Point2d Marker::get_next_position(uint8_t currentFrame) {
		uint8_t frameDiff = currentFrame - mLastFrame;
		return mLastPosition + (mLastVelocity * frameDiff);// +(mLastAcceleration * frameDiff * frameDiff / 2);
	}

	void Marker::set_current_position(uint8_t currentFrame, cv::Point2d* pos) {
		if (pos != nullptr || pos->x < 0 || pos->y < 0) {
			uint8_t frameDiff = currentFrame - mLastFrame;
			if (mLastPosition.x != 0 && mLastPosition.y != 0) {
				cv::Point2d newVelocity = (*pos - mLastPosition) / frameDiff;
				// Probably invalid point as it is jumping pretty far from the original spot.
				if (newVelocity.x * frameDiff > 20 || newVelocity.y * frameDiff > 20) {
					return;
				}
				newVelocity.x = std::min(newVelocity.x, 5.0); // Max x velocity -> px per frame
				newVelocity.y = std::min(newVelocity.y, 5.0); // Max x velocity -> px per frame
				// If we have a previous velocity, we calc the acceleration. 
				// First acceleration after tracking is too high.
				if (mLastVelocity.x != 0 || mLastVelocity.y != 0) {
					mLastAcceleration = 2 * (newVelocity - mLastVelocity) / frameDiff;
				}
				else {
					mLastAcceleration = cv::Point2d(0, 0);
				}
				mLastVelocity = newVelocity;
			}
			mLastPosition = *pos;
			mLastFrame = currentFrame;
			mTracked = true;
		}
	}

	// Returns -1 if not found, 0 if found, 1 if previously tracked position still valid.
	int Marker::find_position(cv::Mat& frame, uint8_t currentFrame, cv::Point2d* position) {
		cv::Mat marker_area;
		cv::Mat marker_area_thresh;
		std::vector<std::vector<cv::Point>> cont;
		cv::Moments mu;
		int returnVal = -1;

		int lx = 0, ly = 0;
		// Decrease search area.
		cv::Point2d predicted_center;
		if (is_trackable(currentFrame)) {
			predicted_center = get_next_position(currentFrame);
			int halfW = (int)std::min(200.0, std::max(40.0, abs(predicted_center.x - mLastPosition.x)));
			int halfH = (int)std::min(200.0, std::max(40.0, abs(predicted_center.y - mLastPosition.y)));
			lx = (int)(std::max(0.0, std::min(predicted_center.x, (double)frame.cols) - halfW));
			ly = (int)(std::max(0.0, std::min(predicted_center.y, (double)frame.rows) - halfH));
			int hx = (int)(std::min((double)frame.cols-1, std::max(0.0, predicted_center.x) + halfW));
			int hy = (int)(std::min((double)frame.rows-1, std::max(0.0, predicted_center.y) + halfH));
			
			marker_area = frame(cv::Rect(lx, ly, hx-lx, hy-ly));

			inRange(marker_area, get_marker_color_range_low(), get_marker_color_range_high(), marker_area_thresh);

			cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
			cv::morphologyEx(marker_area_thresh, marker_area_thresh, cv::MORPH_CLOSE, structuringElement);

			// Find the contour, using offset lx, ly to scale points back to real position.
			cv::findContours(marker_area_thresh, cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(lx, ly));
#ifdef DEBUG_IMG
			cv::Mat out;
			frame.copyTo(out);
			cv::rectangle(out, cv::Rect(lx, ly, hx-lx, hy-ly), cv::Scalar(255, 255, 255), 4);
			cv::namedWindow("Marker Search Area", cv::WINDOW_NORMAL);
			cv::resizeWindow("Marker Search Area", 600, (int)(600 * out.rows / out.cols));
			cv::imshow("Marker Search Area", out);
#endif
		}
		else {
			frame.copyTo(marker_area);

			inRange(marker_area, get_marker_color_range_low(), get_marker_color_range_high(), marker_area_thresh);

			// TODO Replace it with something more efficient as on marker_area = frame will be too slow.

			// Find the contour, using offset lx, ly to scale points back to real position.
			cv::findContours(marker_area_thresh, cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(lx, ly));
#ifdef DEBUG_IMG
			cv::namedWindow("Marker Search Area", cv::WINDOW_NORMAL);
			cv::resizeWindow("Marker Search Area", 600, (int)(600 * frame.rows / frame.cols));
			cv::imshow("Marker Search Area", frame);
#endif
		}

#ifdef DEBUG_IMG
		cv::namedWindow("Marker Threshold Area", cv::WINDOW_NORMAL);
		cv::resizeWindow("Marker Threshold Area", 600, (int)(600 * marker_area_thresh.rows / marker_area_thresh.cols));
#endif

#ifdef DEBUG_IMG
		cv::Mat out;
		if(cont.size() == 0) {
			out = cv::Mat::zeros(cv::Size((int)(100 + 20*mName.length()), 30), CV_8UC3);
			cv::putText(out, "Missing " + mName, cv::Point(5, 20), cv::FONT_HERSHEY_PLAIN, 1, rp::ScalarHSV2BGR(mHsvColor[0], mHsvColor[1], mHsvColor[2]), 1);
		}
		else {
			frame.copyTo(out);
		}
		cv::drawContours(out, cont, -1, cv::Scalar(255, 0, 0), -1);
		cv::namedWindow("Marker Contours", cv::WINDOW_NORMAL);
		cv::resizeWindow("Marker Contours", 600, (int)(600 * out.rows / out.cols));
		cv::imshow("Marker Contours", out);
#endif

		if (cont.size() == 0) {
			// Increase the color range so we hopefully will be able to track it again.
			if (mCHigh[0] - mCLow[0] < (START_H_RANGE_LOW + START_H_RANGE_HIGH)) {
				mCLow[0] = FIX_SUBR_H_RANGE(mCLow[0] - 1);
				mCHigh[0] = FIX_ADD_H_RANGE(mCHigh[0] + 1);
			}
			if (mCHigh[1] - mCLow[1] < (START_S_RANGE_LOW + START_S_RANGE_HIGH)) {
				mCLow[1] = FIX_SUBR_SV_RANGE(mCLow[1], 1);
				mCHigh[1] = FIX_ADD_SV_RANGE(mCHigh[1], 1);
			}
			if (mCHigh[2] - mCLow[2] < (START_V_RANGE_LOW + START_V_RANGE_HIGH)) {
				mCLow[2] = FIX_SUBR_SV_RANGE(mCLow[2], 1);
				mCHigh[2] = FIX_ADD_SV_RANGE(mCHigh[2], 1);
			}
			if (predicted_center != cv::Point2d()) {
				position->x = predicted_center.x;
				position->y = predicted_center.y;
				returnVal = 1;
			}
		}
		else if (cont.size() >= 1) {
			double area;
			double contourX, contourY;
			double contourError;
			double radius;
			double error = INFINITY;
			cv::Mat labels = cv::Mat::zeros(marker_area.size(), CV_8UC1);

			for (int i = 0; i < cont.size(); i++) {
				area = contourArea(cont[i]);
				if (area < pow(mMarkerRadius*2+5, 2)) {
					//mu = moments(cont[i]);
					//if (cont[i].size() == 1 ||mu.m10 < 0 || mu.m01 < 0 || mu.m00 <= 0) {
					auto bb = cv::boundingRect(cont[i]);
					contourX = bb.x + (((double) bb.width) / 2);
					contourY = bb.y + (((double) bb.height) / 2);
					radius = std::max((((double)bb.width) / 2), (((double)bb.height) / 2));
					// If the radius changes too fast, we don't change it.
					if (radius - mMarkerRadius > 3)
						radius = mMarkerRadius;
					//}
					//else {
					//	contourX = std::max(0.0, std::min(mu.m10 / mu.m00, (double)frame.cols));
					//	contourY = std::max(0.0, std::min(mu.m01 / mu.m00, (double)frame.rows));
					//	auto bb = cv::boundingRect(cont[i]);
					//	radius = std::max((((double)bb.width) / 2), (((double)bb.height) / 2));
					//}
					//auto bb = cv::boundingRect(cont[i]);

					// Minimize distance error if we already know it. Otherwise take the one with the smallest color error.
					if (predicted_center != cv::Point2d()) {
						contourError = pow(contourX - predicted_center.x, 2) + pow(contourY - predicted_center.y, 2);
					}
					else {
						cv::drawContours(labels, cont, i, cv::Scalar(i), cv::FILLED);
						cv::Rect roi = cv::boundingRect(cont[i]);
						cv::Scalar mean = cv::mean(marker_area(roi), labels(roi) == i);
						
						contourError = pow(mean[0] - mHsvColor[0], 2) + pow(mean[1] - mHsvColor[1], 2) + pow(mean[2] - mHsvColor[2], 2);
					}
					if (contourError < error) {
						error = contourError;
						position->x = contourX;
						position->y = contourY;
						mMarkerRadius = std::max(1.0, radius);
						returnVal = 0;
					}
				}
			}
#ifdef DEBUG_IMG
			cv::circle(marker_area_thresh, cv::Point((int) position->x - lx, (int) position->y - ly), 1, cv::Scalar(128), -1);
#endif
		}
#ifdef DEBUG_IMG
		cv::imshow("Marker Threshold Area", marker_area_thresh);
		cv::waitKey(0);
		/*cv::destroyWindow("Marker Search Area");
		cv::destroyWindow("Marker Threshold Area");
		cv::destroyWindow("Marker Contours");*/
#endif
		return returnVal;
	}
	
	cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
		cv::Mat rgb;
		cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
		cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
		return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
	}
}

