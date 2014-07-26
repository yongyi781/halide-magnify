// HalideApps.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Util.h"
#include "VideoApp.h"
#include "NamedWindow.h"
#include "EulerianMagnifier.h"
#include "RieszMagnifier.h"

using namespace Halide;

#pragma region Declarations

Var x("x"), y("y"), c("c"), w("w");

// Returns timing in milliseconds.
template<typename F0>
double timing(F0 f, int iterations = 1)
{
	auto t0 = currentTime();
	for (int i = 0; i < iterations; ++i)
		f();
	auto d = currentTime() - t0;
	return d / iterations;
}

// Prints and returns timing in milliseconds
template<typename F0>
double printTiming(F0 f, std::string message = "", int iterations = 1)
{
	if (!message.empty())
		std::cout << message << flush;
	double t = timing(f, iterations);
	std::cout << t << " ms" << std::endl;
	return t;
}

// Converts a Mat to an Image<uint8_t> (channels, width, height).
// Different order: channels = extent(0), width = extent(1), height = extent(2).
Image<uint8_t> toImage_uint8(cv::Mat mat)
{
	return Image<uint8_t>(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));;
}

// Converts a Mat to an Image<uint8_t> and reorders the data to be in the order (width, height, channels).
Image<uint8_t> toImage_uint8_reorder(cv::Mat mat)
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x, y);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));
	return convert.realize(mat.cols, mat.rows, mat.channels());
}

// Converts an Image<uint8_t> (channels, width, height) to a Mat.
cv::Mat toMat(Image<uint8_t> im)
{
	return cv::Mat(im.extent(2), im.extent(1), CV_8UC3, im.data());
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat.
void toMat_reordered(Image<uint8_t> im, cv::Mat mat)
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(c, x, y) = ip(x, y, c);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
}

// Converts a Mat to an Image<float> and reorders the data to be in the order (width, height, channels).
Image<float> toImage_reorder(cv::Mat mat)
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x, y) / 255.0f;
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));
	return convert.realize(mat.cols, mat.rows, mat.channels());
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat (CV_8UC3).
void toMat_reordered(Image<float> im, cv::Mat mat)
{
	static Func convert;
	static ImageParam ip(Float(32), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(c, x, y) = cast<uint8_t>(ip(x, y, c) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat (CV_8UC3).
cv::Mat toMat_reordered(Image<float> im)
{
	static Func convert;
	static ImageParam ip(Float(32), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(c, x, y) = cast<uint8_t>(ip(x, y, c) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC3, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	return mat;
}

#pragma endregion

int main_magnify()
{
	VideoApp app;
	EulerianMagnifier magnifier(app);

	NamedWindow inputWindow("Input"), resultWindow("Result");
	inputWindow.move(0, 0);
	resultWindow.move(app.width() + 10, 0);
	Image<float> frame;
	Image<float> out(app.width(), app.height(), app.channels());
	double timeSum = 0;
	int frameCounter = -10;
	for (int i = 0;; i++, frameCounter++)
	{
		frame = app.readFrame();
		if (frame.dimensions() == 0)
			break;
		double t = currentTime();
		// --- timing ---
		magnifier.process(frame, out);
		// --- end timing ---
		double diff = currentTime() - t;
		inputWindow.showImage(frame);
		resultWindow.showImage(out);
		std::cout << diff << " ms";

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			std::cout << "\t(" << (frameCounter + 1) / timeSum << " FPS)" << std::endl;
		}
		else
		{
			std::cout << std::endl;
		}
		if (cv::waitKey(30) >= 0)
			break;
	}

	return 0;
}

int webcam_control()
{
	NamedWindow window;
	cv::VideoCapture cap(0);

	while (true)
	{
		cv::Mat frame;
		cap >> frame;
		window.showImage(frame);
		if (cv::waitKey(30) >= 0)
			break;
	}

	return 0;
}

int main(int argc, TCHAR* argv[])
{
	return main_magnify();

	//const int N = 50, WIDTH = 256, HEIGHT = 256;

	//Param<float> tParam;
	//Func gen;
	//gen(x, y) = 0.5f + 0.5f * cos(0.05f * (x - sin(0.2f * tParam)) + 0.09f * y);
	//Image<float> in[N];
	//for (int i = 0; i < N; i++)
	//{
	//	tParam.set((float)i);
	//	in[i] = gen.realize(WIDTH, HEIGHT);
	//}

	//NamedWindow window("Input");
	//for (int i = 0;; i++)
	//{
	//	window.showImage(in[i % N]);
	//	if (cv::waitKey(20) >= 0)
	//		break;
	//}
}
