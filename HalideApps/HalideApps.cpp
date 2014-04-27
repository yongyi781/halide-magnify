// HalideApps.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "WebcamApp.h"

using namespace std;
using namespace Halide;
using namespace cv;

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
double printTiming(F0 f, string message = "", int iterations = 1)
{
	if (!message.empty())
		cout << message << flush;
	double t = timing(f, iterations);
	cout << t << " ms" << endl;
	return t;
}

// Clips image access to edges.
Func clipToEdges(const ImageParam& ip)
{
	Func f("clipToEdges");
	f(x, y, _) = ip(clamp(x, 0, ip.width() - 1), clamp(y, 0, ip.height() - 1), _);
	return f;
}

// Downsample with a 1 3 3 1 filter
template<typename F>
Func downsample(F f)
{
	Func downx, downy;

	downx(x, y, _) = (f(2 * x - 1, y, _) + 3.0f * (f(2 * x, y, _) + f(2 * x + 1, y, _)) + f(2 * x + 2, y, _)) / 8.0f;
	downy(x, y, _) = (downx(x, 2 * y - 1, _) + 3.0f * (downx(x, 2 * y, _) + downx(x, 2 * y + 1, _)) + downx(x, 2 * y + 2, _)) / 8.0f;

	return downy;
}

// Upsample using bilinear interpolation
template<typename F>
Func upsample(F f)
{
	Func upx("upx"), upy("upy");

	upx(x, y, _) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y, _) + 0.75f * f(x / 2, y, _);
	upy(x, y, _) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2), _) + 0.75f * upx(x, y / 2, _);

	return upy;
}

// Converts a Mat to an Image<uint8_t> (channels, width, height).
// Different order: channels = extent(0), width = extent(1), height = extent(2).
Image<uint8_t> toImage_uint8(const Mat& mat)
{
	return Image<uint8_t>(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));;
}

// Converts a Mat to an Image<uint8_t> and reorders the data to be in the order (width, height, channels).
Image<uint8_t> toImage_uint8_reorder(const Mat& mat)
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
Mat toMat(const Image<uint8_t>& im)
{
	return Mat(im.extent(2), im.extent(1), CV_8UC3, im.data());
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat.
void toMat_reordered(const Image<uint8_t>& im, Mat& mat)
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
Image<float> toImage_reorder(const Mat& mat)
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
void toMat_reordered(const Image<float>& im, Mat& mat)
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
Mat toMat_reordered(const Image<float>& im)
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
	Mat mat(im.height(), im.width(), CV_8UC3, Scalar(0));
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	return mat;
}

#pragma region Externs

map<int, Image<float>> pyramid; // global result

extern "C" __declspec(dllexport) int copyPyramidLevel(int level, buffer_t *in, buffer_t* out)
{
	if (!pyramid.count(level))
		pyramid[level] = Image<float>(640, 480); // allocate on first attempt to copy data from this level

	if (in->host == nullptr)
	{
		// Bounds inference mode, I guess
		for (int i = 0; i < 2; i++)
		{
			in->min[i] = out->min[i];
			in->extent[i] = out->extent[i];
		}
	}
	else
	{
		// memcpy data from in into pyramid[level] buffer at right location
		int32_t min = in->min[0] * in->stride[0] + in->min[1] * in->stride[1];
		uint8_t* outData = (uint8_t*)pyramid[level].data() + min;
		uint8_t* inData = in->host + min;
		int32_t extent = in->extent[0] * in->stride[0] + in->extent[1] * in->stride[1];
		memcpy(outData, inData, extent);
		memcpy(out->host + min, inData, extent);
	}
	return 0;
}

#pragma endregion

// Returns Gaussian pyramid of an image.
template<int J>
array<Func, J> gaussianPyramid(Func in)
{
	array<Func, J> gPyramid;
	gPyramid[0](x, y, _) = in(x, y, _);
	for (int j = 1; j < J; j++)
		gPyramid[j](x, y, _) = downsample(gPyramid[j - 1])(x, y, _);
	return gPyramid;
}

// Returns Laplacian pyramid of a Gaussian pyramid.
template<typename F, int J>
array<Func, J> laplacianPyramid(array<F, J> gPyramid)
{
	array<Func, J> lPyramid;
	lPyramid[J - 1](x, y, _) = gPyramid[J - 1](x, y, _);
	for (int j = J - 2; j >= 0; j--)
		lPyramid[j](x, y, _) = gPyramid[j](x, y, _) - upsample(gPyramid[j + 1])(x, y, _);
	return lPyramid;
}

// Returns Gaussian pyramid of an input image, as an image.
template<int J>
array<Image<float>, J> gaussianPyramidImages(const Image<float>& in)
{
	static ImageParam prevPyramidInput(Float(32), 2);
	static Func prevPyramidInputClamped;
	static Func pyramidLevel;

	if (!prevPyramidInputClamped.defined() && !pyramidLevel.defined())
	{
		prevPyramidInputClamped = clipToEdges(prevPyramidInput);
		pyramidLevel(x, y) = downsample(prevPyramidInputClamped)(x, y);
	}

	array<Image<float>, 8> gPyramid;
	gPyramid[0] = in;
	for (int j = 1, w = in.width() / 2, h = in.height() / 2; j < J; j++, w /= 2, h /= 2)
	{
		prevPyramidInput.set(gPyramid[j - 1]);
		gPyramid[j] = pyramidLevel.realize(w, h);
	}

	return gPyramid;
}

// Pyramid stored in pyramid.
template<int J>
void gaussianPyramidExtern(const Func in)
{
	Func pyramidLevel[J];
	Func copyLevel[J];
	pyramidLevel[0] = in;
	for (int j = 0; j < J; j++)
	{
		if (j > 0)
			pyramidLevel[j](x, y) = downsample(copyLevel[j])(x, y);
		copyLevel[j].define_extern("copyPyramidLevel", vector < ExternFuncArgument > {level, pyramidLevel[j]}, Int(32), 2);
	}
}

// Reconstructs image from Laplacian pyramid
template<int J>
Func reconstruct(ImageParam(&lPyramid)[J])
{
	Func clamped[J];
	for (int i = 0; i < J; i++)
		clamped[i] = clipToEdges(lPyramid[i]);
	Func output[J];
	output[J - 1](x, y, _) = clamped[J - 1](x, y, _);
	for (int j = J - 2; j >= 0; j--)
		output[j](x, y, _) = upsample(output[j + 1])(x, y, _) + clamped[j](x, y, _);
	for (int i = 1; i < J; i++)
		output[i].compute_root().vectorize(x, 4).parallel(y, 4);
	return output[0];
}

// Sets an array of ImageParams, with offset such that ipArray[i] <- images[(i + offset) % P];
template<int P>
void setImages(ImageParam(&ipArray)[P], Image<float>(&images)[P], int offset = 0)
{
	for (int i = 0; i < P; i++)
		ipArray[i].set(images[(i + offset) % P]);
}

// Sets an array of ImageParams, with offset such that ipArray[i] <- images[(i + offset) % P];
template<int P>
void setImages(array<ImageParam, P>& ipArray, const array<Image<float>, P>& images, int offset = 0)
{
	for (int i = 0; i < P; i++)
		ipArray[i].set(images[(i + offset) % P]);
}

#pragma endregion

int main_v1()
{
	// Number of pyramid levels
	const int J = 8;
	// Number of entries in circular buffer.
	const int P = 5;
	const float alphaValues[J] = { 0, 0, 4, 7, 8, 9, 10, 10 };

	// Input image param.
	ImageParam input(Float(32), 3, "input");

	// Ciruclar buffer image params (x, y). [0] is most recent, [4] is least recent.
	ImageParam bufferInput[P];
	for (int i = 0; i < P; i++)
		bufferInput[i] = ImageParam(Float(32), 2);
	ImageParam temporalProcessOutput[P];
	for (int i = 0; i < P; i++)
		temporalProcessOutput[i] = ImageParam(Float(32), 2);
	// Image params for Laplacian reconstruction, which takes in an image param array.
	ImageParam ipArray[J];
	for (int i = 0; i < J; i++)
		ipArray[i] = ImageParam(Float(32), 2);
	Halide::Param<float> alpha;

	// Reconstruction function.
	Func lReconstruct = reconstruct(ipArray);

	// Algorithm
	Func clamped = lambda(x, y, c, input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c));
	Func grey = lambda(x, y, 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2));
	array<Func, J> gPyramid = gaussianPyramid<J>(grey);
	array<Func, J> lPyramid = laplacianPyramid(gPyramid);
	Func temporalProcess;
	temporalProcess(x, y) = 1.1430f * temporalProcessOutput[2](x, y) - 0.4128f * temporalProcessOutput[4](x, y)
		+ 0.6389f * bufferInput[0](x, y) - 1.2779f * bufferInput[2](x, y)
		+ 0.6389f * bufferInput[4](x, y);
	Func outputProcess;
	outputProcess(x, y) = bufferInput[4](x, y) + alpha * temporalProcessOutput[0](x, y);

	// Reconstruction with color.
	Func reconstruction;
	reconstruction(x, y, c) = clamp(lReconstruct(x, y) * clamped(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Scheduling
	Var xi, yi;
	reconstruction.tile(x, y, xi, yi, 160, 24).vectorize(xi, 4).parallel(y);
	lReconstruct.compute_root().vectorize(x, 4).parallel(y, 4);
	grey.compute_root().vectorize(x, 4).parallel(y, 4);
	for (int j = 1; j < 7; j++)
	{
		gPyramid[j].compute_root().parallel(y, 4).vectorize(x, 4);
	}
	for (int j = 7; j < J; j++)
	{
		gPyramid[j].compute_root();
	}

	WebcamApp app;
	Image<float> pyramidBuffer[P][J];
	Image<float> outputBuffer[P][J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		for (int i = 0; i < P; i++)
			outputBuffer[i][j] = Image<float>(w, h);
	Image<float> outPyramid[J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		outPyramid[j] = Image<float>(w, h);
	double timeSum = 0;
	int frameCounter = -10;
	for (int i = 0;; i++, frameCounter++)
	{
		auto im = app.readFrame();
		double t0 = currentTime();
		// --- timing ---
		input.set(im);
		for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		{
			pyramidBuffer[i % P][j] = lPyramid[j].realize(w, h);
			if (alphaValues[j] == 0.0f || i < P - 1)
			{
				outPyramid[j] = pyramidBuffer[i % P][j];
			}
			else
			{
				for (int p = 0; p < P; p++)
				{
					bufferInput[p].set(pyramidBuffer[(i - p) % P][j]);
					temporalProcessOutput[p].set(outputBuffer[(i - p) % P][j]);
				}
				outputBuffer[i % P][j] = temporalProcess.realize(w, h);
				temporalProcessOutput[0].set(outputBuffer[i % P][j]);
				alpha.set(alphaValues[j]);
				outPyramid[j] = outputProcess.realize(w, h);
			}
		}
		setImages(ipArray, outPyramid);
		Image<float> out = reconstruction.realize(app.width(), app.height(), app.channels());
		// --- end timing ---
		double diff = currentTime() - t0;
		app.showImage(out);
		if (waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			cout << "(" << (frameCounter + 1) / timeSum << " FPS)" << endl;
		}
	}
	cout << "\nAverage FPS: " << frameCounter / timeSum << endl
		<< "Number of frames: " << frameCounter << endl;
	return 0;
}

int main_v2()
{
	const int J = 8;
	const int P = 5;
	const float alphaValues[J] = { 0, 0, 4, 7, 8, 9, 10, 10 };

	WebcamApp app;

	ImageParam input(Float(32), 3, "input");
	Func clamped = lambda(x, y, c, input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c));
	Func grey = lambda(x, y, 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2));

	// Pyramids
	array<ImageParam, J> gPyramidInput;
	for (int j = 0; j < J; j++)
		gPyramidInput[j] = ImageParam(Float(32), 2);
	array<Func, J> gPyramidInputClamped;
	for (int j = 0; j < J; j++)
		gPyramidInputClamped[j] = clipToEdges(gPyramidInput[j]);
	array<Func, J> lPyramid = laplacianPyramid(gPyramidInputClamped);

	// Ciruclar buffer image params (x, y). [0] is most recent, [4] is least recent.
	ImageParam bufferInput[P];
	for (int i = 0; i < P; i++)
		bufferInput[i] = ImageParam(Float(32), 2);
	ImageParam temporalProcessOutput[P];
	for (int i = 0; i < P; i++)
		temporalProcessOutput[i] = ImageParam(Float(32), 2);

	// Image params for Laplacian reconstruction, which takes in an image param array.
	ImageParam ipArray[J];
	for (int i = 0; i < J; i++)
		ipArray[i] = ImageParam(Float(32), 2);
	Halide::Param<float> alpha;

	// Reconstruction function.
	Func lReconstruct = reconstruct(ipArray);

	// Algorithm
	Func temporalProcess;
	temporalProcess(x, y) = 1.1430f * temporalProcessOutput[2](x, y) - 0.4128f * temporalProcessOutput[4](x, y)
		+ 0.6389f * bufferInput[0](x, y) - 1.2779f * bufferInput[2](x, y)
		+ 0.6389f * bufferInput[4](x, y);
	Func outputProcess;
	outputProcess(x, y) = bufferInput[4](x, y) + alpha * temporalProcessOutput[0](x, y);

	// Reconstruction with color.
	Func reconstruction;
	reconstruction(x, y, c) = clamp(lReconstruct(x, y) * clamped(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Scheduling
	Var xi, yi;
	reconstruction.tile(x, y, xi, yi, 160, 24).vectorize(xi, 4).parallel(y);
	lReconstruct.compute_root().vectorize(x, 4).parallel(y, 4);
	grey.compute_root().vectorize(x, 4).parallel(y, 4);

	Image<float> pyramidBuffer[P][J];
	Image<float> outputBuffer[P][J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		for (int i = 0; i < P; i++)
			outputBuffer[i][j] = Image<float>(w, h);
	Image<float> outPyramid[J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		outPyramid[j] = Image<float>(w, h);
	double timeSum = 0;
	int frameCounter = -10;

	// Main loop
	for (int i = 0;; i++, frameCounter++)
	{
		Image<float> frame;
		Image<float> out;
		frame = app.readFrame();
		double t = currentTime();
		// --- timing ---
		input.set(frame);
		auto gImages = gaussianPyramidImages<J>(grey.realize(app.width(), app.height()));
		setImages(gPyramidInput, gImages);
		for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		{
			pyramidBuffer[i % P][j] = lPyramid[j].realize(w, h);
			if (alphaValues[j] == 0.0f || i < P - 1)
			{
				outPyramid[j] = pyramidBuffer[i % P][j];
			}
			else
			{
				for (int p = 0; p < P; p++)
				{
					bufferInput[p].set(pyramidBuffer[(i - p) % P][j]);
					temporalProcessOutput[p].set(outputBuffer[(i - p) % P][j]);
				}
				outputBuffer[i % P][j] = temporalProcess.realize(w, h);
				temporalProcessOutput[0].set(outputBuffer[i % P][j]);
				alpha.set(alphaValues[j]);
				outPyramid[j] = outputProcess.realize(w, h);
			}
		}
		setImages(ipArray, outPyramid);
		out = reconstruction.realize(app.width(), app.height(), app.channels());
		// --- end timing ---
		double diff = currentTime() - t;
		app.showImage(out);
		cout << diff << " ms";
		if (waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			cout << "\t(" << (frameCounter + 1) / timeSum << " FPS)" << endl;
		}
		else
		{
			cout << endl;
		}
	}

	return 0;
}

int main_v3()
{
	const int J = 8;
	const int P = 5;
	const float alphaValues[J] = { 0, 0, 4, 7, 8, 9, 10, 10 };

	WebcamApp app;

	ImageParam input(Float(32), 3, "input");
	Func clamped = lambda(x, y, c, input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c));
	Func grey = lambda(x, y, 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2));

	// Pyramids
	array<ImageParam, J> gPyramidInput;
	for (int j = 0; j < J; j++)
		gPyramidInput[j] = ImageParam(Float(32), 2);
	array<Func, J> gPyramidInputClamped;
	for (int j = 0; j < J; j++)
		gPyramidInputClamped[j] = clipToEdges(gPyramidInput[j]);
	array<Func, J> lPyramid = laplacianPyramid(gPyramidInputClamped);

	// Ciruclar buffer image params (x, y). [0] is most recent, [4] is least recent.
	ImageParam bufferInput[P];
	for (int i = 0; i < P; i++)
		bufferInput[i] = ImageParam(Float(32), 2);
	ImageParam temporalProcessOutput[P];
	for (int i = 0; i < P; i++)
		temporalProcessOutput[i] = ImageParam(Float(32), 2);

	// Image params for Laplacian reconstruction, which takes in an image param array.
	ImageParam ipArray[J];
	for (int i = 0; i < J; i++)
		ipArray[i] = ImageParam(Float(32), 2);
	Halide::Param<float> alpha;

	// Reconstruction function.
	Func lReconstruct = reconstruct(ipArray);

	// Algorithm
	Func temporalProcess;
	temporalProcess(x, y) = 1.1430f * temporalProcessOutput[2](x, y) - 0.4128f * temporalProcessOutput[4](x, y)
		+ 0.6389f * bufferInput[0](x, y) - 1.2779f * bufferInput[2](x, y)
		+ 0.6389f * bufferInput[4](x, y);
	Func outputProcess;
	outputProcess(x, y) = bufferInput[4](x, y) + alpha * temporalProcessOutput[0](x, y);

	// Reconstruction with color.
	Func reconstruction;
	reconstruction(x, y, c) = clamp(lReconstruct(x, y) * clamped(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Scheduling
	Var xi, yi;
	reconstruction.tile(x, y, xi, yi, 160, 24).vectorize(xi, 4).parallel(y);
	lReconstruct.compute_root().vectorize(x, 4).parallel(y, 4);
	grey.compute_root().vectorize(x, 4).parallel(y, 4);

	Image<float> pyramidBuffer[P][J];
	Image<float> outputBuffer[P][J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		for (int i = 0; i < P; i++)
			outputBuffer[i][j] = Image<float>(w, h);
	Image<float> outPyramid[J];
	for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		outPyramid[j] = Image<float>(w, h);
	double timeSum = 0;
	int frameCounter = -10;

	// Main loop
	for (int i = 0;; i++, frameCounter++)
	{
		Image<float> frame;
		Image<float> out;
		frame = app.readFrame();
		double t = currentTime();
		// --- timing ---
		input.set(frame);
		auto gImages = gaussianPyramidImages<J>(grey.realize(app.width(), app.height()));
		setImages(gPyramidInput, gImages);
		for (int j = 0, w = app.width(), h = app.height(); j < J; j++, w /= 2, h /= 2)
		{
			pyramidBuffer[i % P][j] = lPyramid[j].realize(w, h);
			if (alphaValues[j] == 0.0f || i < P - 1)
			{
				outPyramid[j] = pyramidBuffer[i % P][j];
			}
			else
			{
				for (int p = 0; p < P; p++)
				{
					bufferInput[p].set(pyramidBuffer[(i - p) % P][j]);
					temporalProcessOutput[p].set(outputBuffer[(i - p) % P][j]);
				}
				outputBuffer[i % P][j] = temporalProcess.realize(w, h);
				temporalProcessOutput[0].set(outputBuffer[i % P][j]);
				alpha.set(alphaValues[j]);
				outPyramid[j] = outputProcess.realize(w, h);
			}
		}
		setImages(ipArray, outPyramid);
		out = reconstruction.realize(app.width(), app.height(), app.channels());
		// --- end timing ---
		double diff = currentTime() - t;
		app.showImage(out);
		cout << diff << " ms";
		if (waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			cout << "\t(" << (frameCounter + 1) / timeSum << " FPS)" << endl;
		}
		else
		{
			cout << endl;
		}
	}

	return 0;
}

map<int, Image<float>> stuff;

int scaleSize(int initialSize, int level)
{
	while (--level > 0)
		initialSize /= 2;
	return initialSize;
}

int aa = 0;

extern "C" __declspec(dllexport) int myExtern(int level, buffer_t *in, buffer_t *out)
{
	if (in->host == nullptr)
	{
		for (int i = 0; i < 2; i++)
		{
			in->min[i] = out->min[i];
			in->extent[i] = out->extent[i];
		}
	}
	else
	{
		if (stuff.count(level) == 0)
			stuff[level] = Image<float>(scaleSize(640, level), scaleSize(480, level));
		int* dst = (int*)out->host;
		int* src = (int*)in->host;
		for (int y = 0; y < out->extent[1]; y++)
		{
			int* dstLine = dst + y * out->stride[1];
			int* srcLine = src + y * in->stride[1];
			for (int x = 0; x < out->extent[0]; x++)
				dstLine[x] = ++aa;
		}
	}
	return 0;
}

int main(int argc, TCHAR* argv[])
{
	Func f("f"), g("g"), h("h");

	f(x, y) = x + y;

	// Name of the function and the args, then types of the outputs, then dimensionality
	Halide::Param<int> param;
	param.set(1);
	g.define_extern("myExtern", vector < ExternFuncArgument > {param, f}, Int(32), 2);

	h(x, y) = g(x, y);

	Var xi, yi;
	f.compute_root();
	h.tile(x, y, xi, yi, 4, 4);
	g.compute_at(h, x);

	Image<int> result = h.realize(20, 20);

	for (int y = 0; y < 20; y++)
	{
		for (int x = 0; x < 20; x++)
			cout << result(x, y) << " ";
		cout << "\n";
	}

	return 0;
}
