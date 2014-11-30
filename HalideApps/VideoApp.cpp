#include "stdafx.h"
#include "VideoApp.h"

using namespace Halide;

VideoApp::VideoApp(int scaleFactor) : scaleFactor(scaleFactor), cap(0)
{
	if (!cap.isOpened())
		throw std::exception("Cannot open webcam.");
}

VideoApp::VideoApp(std::string filename) : scaleFactor(1), cap(filename)
{
	if (!cap.isOpened())
		throw std::exception("Cannot open file.");
}

Image<float> VideoApp::readFrame()
{
	static Func convert("convertFromMat");
	static ImageParam ip(UInt(8), 3);
	static Var x("x"), y("y"), c("c");

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x / scaleFactor, y / scaleFactor) / 255.0f;
		convert.vectorize(x, 4).parallel(y, 4);
	}

	cv::Mat frame;
	cap >> frame;
	if (frame.empty())
		return Image<float>();

	ip.set(Buffer(UInt(8), frame.channels(), frame.cols, frame.rows, 0, frame.data));
	return convert.realize(scaleFactor * frame.cols, scaleFactor * frame.rows, frame.channels());
}

Image<uint8_t> VideoApp::readFrame_uint8()
{
	static Func convert("convertFromMat");
	static ImageParam ip(UInt(8), 3);
	static Var x("x"), y("y"), c("c");

	if (!convert.defined())
	{
		convert(x, y) = ip(0, x / scaleFactor, y / scaleFactor);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	cv::Mat frame;
	cap >> frame;
	if (frame.empty())
		return Image<float>();

	ip.set(Buffer(UInt(8), frame.channels(), frame.cols, frame.rows, 0, frame.data));
	return convert.realize(scaleFactor * frame.cols, scaleFactor * frame.rows);
}
