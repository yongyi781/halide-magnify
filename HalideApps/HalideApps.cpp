// HalideApps.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Util.h"
#include "VideoApp.h"
#include "NamedWindow.h"
#include "EulerianMagnifier.h"
#include "RieszMagnifier.h"
#include "filter_util.h"

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
		std::cout << message << std::flush;
	double t = timing(f, iterations);
	std::cout << t << " ms" << std::endl;
	return t;
}

// Converts a Mat to an Image<uint8_t> (channels, width, height).
// Different order: channels = extent(0), width = extent(1), height = extent(2).
Image<uint8_t> toImage_uint8(const cv::Mat& mat)
{
	return Image<uint8_t>(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));;
}

// Converts a Mat to an Image<uint8_t> and reorders the data to be in the order (width, height, channels).
Image<uint8_t> toImage_uint8_reorder(const cv::Mat& mat)
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

// Converts a Mat to an Image<float> and reorders the data to be in the order (width, height, channels).
Image<float> toImage_reorder(const cv::Mat& mat)
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

// Converts a reordered Image<uint8_t> (width, height) to a Mat (CV_8U).
cv::Mat toMat2d(Image<float> im)
{
	static Func convert;
	static ImageParam ip(Float(32), 2);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y) = cast<uint8_t>(ip(x, y) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8U, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.width(), im.height(), 0, 0, mat.data));
	return mat;
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
	std::string filename = "C:\\Users\\Yongyi\\Documents\\Visual Studio 2013\\Projects\\HalideApps\\HalideApps\\video.avi";
	std::string filename2 = R"(C:\Users\Yongyi\Downloads\Saved\Video Magnification\RieszPyramidICCP2014pres\inputC.wmv)";
	std::string filename3 = R"(C:\Users\Yongyi\Documents\MATLAB\EVM_Matlab\data\baby.avi)";
	RieszMagnifier magnifier(3, Float(32), 1);
	//EulerianMagnifier magnifier(app, 6, { 3.75, 7.5, 15, 30, 30, 30, 30, 30 });
	magnifier.compileJIT(true);

	std::vector<double> filterA;
	std::vector<double> filterB;
	float alpha = 30.0f;
	double fps = 50.0;
	double videoFps = 0;
	double freqCenter = 1;
	double freqWidth = .5;

	VideoApp app;
	videoFps = app.fps();
	filter_util::computeFilter(videoFps == 0 ? fps : videoFps, freqCenter, freqWidth, filterA, filterB);

	std::vector<Image<float>> historyBuffer;
	for (int i = 0; i < magnifier.getPyramidLevels(); i++)
		historyBuffer.push_back(Image<float>(scaleSize(app.width(), i), scaleSize(app.height(), i), 7, 2));
	magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);

	//cv::VideoWriter writer("output.avi", -1, videoFps == 0 ? fps : videoFps, { app.width(), app.height() });
	NamedWindow inputWindow("Input"), resultWindow("Result");
	inputWindow.move(0, 0);
	resultWindow.move(app.width() + 10, 0);
	Image<float> frame;
	Image<float> out(app.width(), app.height(), app.channels());
	double timeSum = 0;
	int frameCounter = -10;
	int pressedKey;
	for (int i = 0;; i++, frameCounter++)
	{
		frame = app.readFrame();
		if (frame.dimensions() == 0)
		{
			cv::waitKey();
			break;
		}

		double t = currentTime();
		// --- timing ---
		magnifier.process(frame, out);
		//std::cout << out(175, 226) << std::endl;
		// --- end timing ---
		double diff = currentTime() - t;
		inputWindow.showImage(frame);
		resultWindow.showImage(out);
		//writer.write(toMat_reordered(out));
		std::cout << diff << " ms";

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			fps = (frameCounter + 1) / timeSum;
			std::cout << "\t(" << fps << " FPS)"
				<< "\t(" << 1000 / fps << " ms)";

			if (frameCounter % 10 == 0)
			{
				// Update fps
				filter_util::computeFilter(videoFps == 0 ? fps : videoFps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
		}
		std::cout << std::endl;

		if ((pressedKey = cv::waitKey(30)) >= 0) {
			std::cout << pressedKey << std::endl;
			if (pressedKey == 45)	// minus
			{
				freqCenter = std::max(freqWidth, freqCenter - 0.5);
				std::cout << "Freq center is now " << freqCenter << std::endl;
				filter_util::computeFilter(videoFps == 0 ? fps : videoFps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 43)	// plus
			{
				freqCenter += 0.5;
				std::cout << "Freq center is now " << freqCenter << std::endl;
				filter_util::computeFilter(videoFps == 0 ? fps : videoFps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 97)	// a
			{
				// Increase alpha
				alpha += 10;
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 27)
				break;
		}
	}

	return 0;
}

int main_synthetic()
{
	const int N = 100, WIDTH = 256, HEIGHT = 256;

	Param<float> tParam;
	Func gen;
	Expr distSquared = (x - WIDTH / 2 - 0.1f * sin(0.2f * tParam))*(x - WIDTH / 2 - 0.1f * sin(0.2f * tParam)) + (y - HEIGHT / 2)*(y - HEIGHT / 2);
	gen(x, y) = select(distSquared <= WIDTH * WIDTH / 16, 1.0f, select(distSquared <= WIDTH * WIDTH / 16 + 100, 0.5f, 0.0f));
	Image<float> in[N];
	for (int i = 0; i < N; i++)
	{
		tParam.set((float)i);
		in[i] = gen.realize(WIDTH, HEIGHT);
	}

	cv::VideoWriter writer("video.avi", -1, 60.0, { WIDTH, HEIGHT });
	for (int i = 0; i < N; i++)
		writer.write(toMat2d(in[i]));
	writer.release();

	//NamedWindow window("Input");
	//for (int i = 0;; i++)
	//{
	//	window.showImage(in[i % N]);
	//	if (cv::waitKey(20) >= 0)
	//		break;
	//}

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

cv::Mat correctGamma(cv::Mat& img, double gamma) {
	double inverse_gamma = 1.0 / gamma;

	cv::Mat lut_matrix(1, 256, CV_8UC1);
	uchar * ptr = lut_matrix.ptr();
	for (int i = 0; i < 256; i++)
		ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

	cv::Mat result;
	LUT(img, lut_matrix, result);

	return result;
}

int _tmain(int argc, TCHAR* argv[])
{
	if (argc >= 2 && argv[1] == std::wstring(L"-c"))
	{
		RieszMagnifier(1, UInt(8), 7).compileToFile("magnify", true, parse_target_string("arm-64-android"));
	}
	else
	{
		return main_magnify();
	}
}
