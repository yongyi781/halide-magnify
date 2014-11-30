#pragma once

class VideoApp
{
public:
	VideoApp(int scaleFactor = 1);
	VideoApp(std::string filename);

	int width() { return scaleFactor * (int)cap.get(cv::CAP_PROP_FRAME_WIDTH); }
	int height() { return scaleFactor * (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT); }
	int channels() const { return 3; }
	double fps() { return cap.get(cv::CAP_PROP_FPS); }

	Halide::Image<float> readFrame();
	Halide::Image<uint8_t> readFrame_uint8();

private:
	int scaleFactor;
	cv::VideoCapture cap;
};
