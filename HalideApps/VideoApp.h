#pragma once

class VideoApp
{
public:
	VideoApp(int scaleFactor = 1);
	VideoApp(std::string filename);

	int width() { return scaleFactor * (int)cap.get(CV_CAP_PROP_FRAME_WIDTH); }
	int height() { return scaleFactor * (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT); }
	int channels() const { return 3; }

	Halide::Image<float> readFrame();

private:
	int scaleFactor;
	cv::VideoCapture cap;
};
