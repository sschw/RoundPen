#include "Background.h"

namespace rp {
#define FIX_SUBR_H_RANGE(x) (((x) > 180) ? x - 76 : x)
#define FIX_ADD_H_RANGE(x) (((x) > 180) ? x + 76 : x)

	const int MINIMAL_HSV_RANGE = 15;
	const int START_HSV_RANGE = 25;

	// If marker color is unique, threshold will only use H channel.
	// If marker saturation is unique, threshold will use HS channel.
	// If marker brightness is unique, threshold will use HSV channel.
	void Background::calibrate_background_range(cv::Mat& img) {
		// Define minimal and maximal range.
		mCLow = cv::Vec3b(FIX_SUBR_H_RANGE(mHsvColor[0] - START_HSV_RANGE - 1), FIX_SUBR_H_RANGE(mHsvColor[1] - START_HSV_RANGE - 1), FIX_SUBR_H_RANGE(mHsvColor[2] - START_HSV_RANGE - 1));
		mCHigh = cv::Vec3b(FIX_ADD_H_RANGE(mHsvColor[0] + START_HSV_RANGE + 1), FIX_ADD_H_RANGE(mHsvColor[1] + START_HSV_RANGE + 1), FIX_ADD_H_RANGE(mHsvColor[2] + START_HSV_RANGE + 1));
		// We work on tmp.
		cv::Mat tmp;
		img.copyTo(tmp);
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		int i = 0;
		// Optimize H until good enough or impossible.
		do {
			mCLow[0] = FIX_ADD_H_RANGE(mCLow[0] + 1);
			mCHigh[0] = FIX_SUBR_H_RANGE(mCHigh[0] - 1);

			cv::inRange(tmp, mCLow, mCHigh, tmp);

			cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

			i++;
		} while (i < START_HSV_RANGE - MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) > 2000));

		// If no good optimization, optimize L until good enough or impossible.
		if (i == START_HSV_RANGE - MINIMAL_HSV_RANGE) {
			i = 0;
			do {
				mCLow[1] = mCLow[1] + 1;
				mCHigh[1] = mCHigh[1] - 1;

				cv::inRange(tmp, mCLow, mCHigh, tmp);

				cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

				i++;
			} while (i < START_HSV_RANGE - MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) > 2000));

			// If no good optimization, optimize S until good enough.
			if (i == START_HSV_RANGE - MINIMAL_HSV_RANGE) {
				i = 0;
				do {
					mCLow[2] = mCLow[2] + 1;
					mCHigh[2] = mCHigh[2] - 1;

					cv::inRange(tmp, mCLow, mCHigh, tmp);

					cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

					i++;
				} while (i < START_HSV_RANGE - MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) > 2000));
			}
		}

		// Range optimized on first frame.
		// Program itself has better detection as too large areas will just be ignored.
	}

	cv::Vec3b Background::get_background_color() {
		return mHsvColor;
	}

	cv::Vec3b Background::get_background_color_range_low() {
		return mCLow;
	}

	cv::Vec3b Background::get_background_color_range_high() {
		return mCHigh;
	}
}