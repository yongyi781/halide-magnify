// HalideApps.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "WebcamApp.h"
#include "NamedWindow.h"

using namespace Halide;

#pragma region Declarations

Var x("x"), y("y"), c("c"), w("w");

// Returns initialSize / 2^level. Used for pyramids.
int scaleSize(int initialSize, int level)
{
	while (--level >= 0)
		initialSize /= 2;
	return initialSize;
}

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
	Func downx("downx"), downy("downy");

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
cv::Mat toMat(const Image<uint8_t>& im)
{
	return cv::Mat(im.extent(2), im.extent(1), CV_8UC3, im.data());
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat.
void toMat_reordered(const Image<uint8_t>& im, cv::Mat& mat)
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

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat (CV_8UC3).
void toMat_reordered(const Image<float>& im, cv::Mat& mat)
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
cv::Mat toMat_reordered(const Image<float>& im)
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

// Returns Gaussian pyramid of an image.
template<int J>
std::array<Func, J> gaussianPyramid(Func in)
{
	std::array<Func, J> gPyramid;
	gPyramid[0](x, y, _) = in(x, y, _);
	for (int j = 1; j < J; j++)
		gPyramid[j](x, y, _) = downsample(gPyramid[j - 1])(x, y, _);
	return gPyramid;
}

// Returns Laplacian pyramid of a Gaussian pyramid.
template<typename F, int J>
std::array<Func, J> laplacianPyramid(std::array<F, J> gPyramid)
{
	std::array<Func, J> lPyramid;
	lPyramid[J - 1](x, y, _) = gPyramid[J - 1](x, y, _);
	for (int j = J - 2; j >= 0; j--)
		lPyramid[j](x, y, _) = gPyramid[j](x, y, _) - upsample(gPyramid[j + 1])(x, y, _);
	return lPyramid;
}

// Returns Gaussian pyramid of an input image, as an image.
template<int J>
std::array<Image<float>, J> gaussianPyramidImages(const Image<float>& in)
{
	static ImageParam prevPyramidInput(Float(32), 2);
	static Func prevPyramidInputClamped;
	static Func pyramidLevel;

	if (!prevPyramidInputClamped.defined() && !pyramidLevel.defined())
	{
		prevPyramidInputClamped = clipToEdges(prevPyramidInput);
		pyramidLevel(x, y) = downsample(prevPyramidInputClamped)(x, y);
	}

	std::array<Image<float>, 8> gPyramid;
	gPyramid[0] = in;
	for (int j = 1, w = in.width() / 2, h = in.height() / 2; j < J; j++, w /= 2, h /= 2)
	{
		prevPyramidInput.set(gPyramid[j - 1]);
		gPyramid[j] = pyramidLevel.realize(w, h);
	}

	return gPyramid;
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
void setImages(std::array<ImageParam, P>& ipArray, const std::array<Image<float>, P>& images, int offset = 0)
{
	for (int i = 0; i < P; i++)
		ipArray[i].set(images[(i + offset) % P]);
}

#pragma endregion

// First version of algorithm.
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
	Param<float> alpha;

	// Reconstruction function.
	Func lReconstruct = reconstruct(ipArray);

	// Algorithm
	Func clamped = lambda(x, y, c, input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c));
	Func grey = lambda(x, y, 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2));
	std::array<Func, J> gPyramid = gaussianPyramid<J>(grey);
	std::array<Func, J> lPyramid = laplacianPyramid(gPyramid);
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
	NamedWindow window;
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
		window.showImage(out);
		if (cv::waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			std::cout << "(" << (frameCounter + 1) / timeSum << " FPS)" << std::endl;
		}
	}
	std::cout << "\nAverage FPS: " << frameCounter / timeSum << std::endl
		<< "Number of frames: " << frameCounter << std::endl;
	return 0;
}

// Full algorithm with intermediate realizing (i.e. not one pipeline)
int main_v2()
{
	const int J = 8;
	const int P = 5;
	const float alphaValues[J] = { 0, 0, 4, 7, 8, 9, 10, 10 };

	ImageParam input(Float(32), 3, "input");
	Func clamped = lambda(x, y, c, input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c));
	Func grey = lambda(x, y, 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2));

	// Pyramids
	std::array<ImageParam, J> gPyramidInput;
	for (int j = 0; j < J; j++)
		gPyramidInput[j] = ImageParam(Float(32), 2);
	std::array<Func, J> gPyramidInputClamped;
	for (int j = 0; j < J; j++)
		gPyramidInputClamped[j] = clipToEdges(gPyramidInput[j]);
	std::array<Func, J> lPyramid = laplacianPyramid(gPyramidInputClamped);

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
	Param<float> alpha;

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

	WebcamApp app;
	NamedWindow window;
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
		window.showImage(out);
		std::cout << diff << " ms";
		if (cv::waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			std::cout << "\t(" << (frameCounter + 1) / timeSum << " FPS)" << std::endl;
		}
		else
		{
			std::cout << std::endl;
		}
	}

	return 0;
}

// Number of pyramid levels
const int PYRAMID_LEVELS = 8;
// Size of circular buffer
const int CIRCBUFFER_SIZE = 5;

std::array<Image<float>, PYRAMID_LEVELS> pyramidBuffer;
std::array<Image<float>, PYRAMID_LEVELS> temporalOutBuffer;

#define TRACE 0

// Extern function to copy data to an external pointer.
extern "C" __declspec(dllexport) int copyFloat32(int p, int j, bool copyToTemporalOut, buffer_t *in, buffer_t *out)
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
#if TRACE
		printf("Copy(%d, %d, %d) called over [%d, %d] x [%d, %d]\n", p, j, copyToTemporalOut, out->min[0], out->min[0] + out->extent[0], out->min[1], out->min[1] + out->extent[1]);
#endif
		float* src = (float*)in->host;
		float* dst = (float*)out->host;
		float* data = copyToTemporalOut ? temporalOutBuffer[j].data() + p * temporalOutBuffer[j].stride(2) : pyramidBuffer[j].data() + p * pyramidBuffer[j].stride(2);
		for (int y = out->min[1]; y < out->min[1] + out->extent[1]; y++)
		{
			float* srcLine = src + (y - in->min[1]) * in->stride[1];
			float* dstLine = dst + (y - out->min[1]) * out->stride[1];
			float* levelLine = data + (y - in->min[1]) * in->stride[1];
			memcpy(dstLine, srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
			memcpy(levelLine, srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
		}
	}
	return 0;
}

Func clipToEdges(Image<float> im)
{
	return lambda(x, y, _, im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1), _));
}

Func clipToEdges(Func f, int width, int height)
{
	return lambda(x, y, _, f(clamp(x, 0, width - 1), clamp(y, 0, height - 1), _));
}

// Full algorithm with one pipeline.
int main_v3()
{
	const float alphaValues[PYRAMID_LEVELS] = { 0, 0, 4, 7, 8, 9, 10, 10 };
	Param<int> pParam;

	WebcamApp app;
	ImageParam input(Float(32), 3);
	// Initialize pyramid buffers
	for (int p = 0; p < CIRCBUFFER_SIZE; p++)
	{
		for (int j = 0; j < PYRAMID_LEVELS; j++)
		{
			pyramidBuffer[j] = Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE);
			temporalOutBuffer[j] = Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE);
		}
	}

	Func clamped("clamped"); clamped(x, y, c) = clipToEdges(input)(x, y, c);
	Func grey("grey"); grey(x, y) = 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2);

	// Gaussian pyramid
	Func gPyramid[PYRAMID_LEVELS];
	gPyramid[0] = Func("gPyramid0");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j < PYRAMID_LEVELS; j++)
	{
		gPyramid[j] = Func("gPyramid" + std::to_string(j));
		gPyramid[j](x, y) = downsample(gPyramid[j - 1])(x, y);
	}

	// Laplacian pyramid
	Func lPyramid[PYRAMID_LEVELS];
	lPyramid[PYRAMID_LEVELS - 1] = Func("lPyramid" + std::to_string(PYRAMID_LEVELS - 1));
	lPyramid[PYRAMID_LEVELS - 1](x, y) = gPyramid[PYRAMID_LEVELS - 1](x, y);
	for (int j = PYRAMID_LEVELS - 2; j >= 0; j--)
	{
		lPyramid[j] = Func("lPyramid" + std::to_string(j));
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(gPyramid[j + 1])(x, y);
	}

	// Copy to pyramid buffer
	Func lPyramidWithCopy[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		Param<int> jParam;
		Param<bool> copyToTemporalOutput;
		jParam.set(j);
		copyToTemporalOutput.set(false);
		lPyramidWithCopy[j] = Func("lPyramidWithCopy" + std::to_string(j));
		lPyramidWithCopy[j].define_extern("copyFloat32", { pParam, jParam, copyToTemporalOutput, lPyramid[j] }, Float(32), 2);
	}

	Func temporalProcess[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		temporalProcess[j] = Func("temporalProcess" + std::to_string(j));
		temporalProcess[j](x, y) =
			1.1430f * temporalOutBuffer[j](x, y, (pParam - 2 + 5) % 5)
			- 0.4128f * temporalOutBuffer[j](x, y, (pParam - 4 + 5) % 5)
			+ 0.6389f * lPyramidWithCopy[j](x, y)
			- 1.2779f * pyramidBuffer[j](x, y, (pParam - 2 + 5) % 5)
			+ 0.6389f * pyramidBuffer[j](x, y, (pParam - 4 + 5) % 5);
	}

	Func temporalProcessWithCopy[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		Param<int> jParam;
		Param<bool> copyToTemporalOutput;
		jParam.set(j);
		copyToTemporalOutput.set(true);
		temporalProcessWithCopy[j] = Func("temporalProcessWithCopy" + std::to_string(j));
		temporalProcessWithCopy[j].define_extern("copyFloat32", { pParam, jParam, copyToTemporalOutput, temporalProcess[j] }, Float(32), 2);
	}

	Func outLPyramid[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		outLPyramid[j] = Func("outLPyramid" + std::to_string(j));
		outLPyramid[j](x, y) = clipToEdges((Func)(lPyramid[j] + (alphaValues[j] == 0.0f ? 0.0f : alphaValues[j] * temporalProcessWithCopy[j])), scaleSize(app.width(), j), scaleSize(app.height(), j))(x, y);
	}

	Func outGPyramid[PYRAMID_LEVELS];
	outGPyramid[PYRAMID_LEVELS - 1](x, y) = outLPyramid[PYRAMID_LEVELS - 1](x, y);
	for (int j = PYRAMID_LEVELS - 2; j >= 0; j--)
	{
		outGPyramid[j] = Func("outGPyramid" + std::to_string(j));
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsample(outGPyramid[j + 1])(x, y);
	}

	Func output("output");
	output(x, y, c) = clamp(outGPyramid[0](x, y) * clamped(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Schedule
	grey.compute_root().parallel(y, 4).vectorize(x, 4);

	Var xi, yi;
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		lPyramid[j].compute_root();
		lPyramidWithCopy[j].compute_root();
		temporalProcess[j].compute_root();
		temporalProcessWithCopy[j].compute_root();
		if (j > 0)
		{
			gPyramid[j].compute_root();
			outGPyramid[j].compute_root();
		}

		if (j <= 4)
		{
			lPyramid[j].parallel(y, 4).vectorize(x, 4);
			if (j > 0)
			{
				gPyramid[j].parallel(y, 4).vectorize(x, 4);
				outGPyramid[j].parallel(y, 4).vectorize(x, 4);
			}
		}
		else
		{
			lPyramid[j].parallel(y).vectorize(x, 4);
			if (j > 0)
			{
				gPyramid[j].parallel(y).vectorize(x, 4);
				outGPyramid[j].parallel(y).vectorize(x, 4);
			}
		}
	}

	output.tile(x, y, xi, yi, 32, 4).parallel(y, 4).vectorize(x, 4);

	// Compile
	std::cout << "Compiling...";
	output.compile_jit();
	for (int j = 0; j < PYRAMID_LEVELS; j++)
		lPyramidWithCopy[j].compile_jit();
	std::cout << "\nDone compiling!\n";

	NamedWindow window("Results");
	window.resize(640, 480);
	double timeSum = 0;
	int frameCounter = -10;
	for (int i = 0;; i++, frameCounter++)
	{
		Image<float> frame = app.readFrame();
		int p = i % CIRCBUFFER_SIZE;
		pParam.set(p);
		input.set(frame);

		if (i < CIRCBUFFER_SIZE - 1)
		{
			for (int j = 0; j < PYRAMID_LEVELS; j++)
			{
				lPyramidWithCopy[j].realize(scaleSize(app.width(), j), scaleSize(app.height(), j));
			}
		}
		else
		{
			double t = currentTime();
			// --- timing ---
			Image<float> out = output.realize(app.width(), app.height(), app.channels());
			// --- end timing ---
			double diff = currentTime() - t;
			window.showImage(out);
			std::cout << diff << " ms";
			if (cv::waitKey(30) >= 0)
				break;

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
	return main_v3();

	//const int J = 6;
	//const int SIZE = 80;

	//Func f("f");

	//f(x, y) = cast<float>(x + 2 * y);

	//for (int level = 0; level < 8; level++)
	//	stuff[level] = Image<float>(scaleSize(SIZE, level), scaleSize(SIZE, level));

	//Func g[J], h[J];
	//for (int j = 0; j < J; j++)
	//{
	//	Param<float*> param;
	//	param.set(stuff[j].data());
	//	g[j].define_extern("copyFloat32", vector < ExternFuncArgument > {param, f}, Float(32), 2);
	//	h[j](x, y) = g[j](x, y);
	//}

	//Var xi, yi;
	//for (int j = 0; j < J; j++)
	//{
	//	f.compute_root();
	//	g[j].compute_at(h[j], x);
	//	if (scaleSize(SIZE, j) % 4 == 0)
	//		h[j].tile(x, y, xi, yi, 4, 4).vectorize(x, 4).parallel(y, 4);
	//}

	//Image<float> result[J];
	//for (int j = 0; j < J; j++)
	//	result[j] = h[j].realize(scaleSize(SIZE, j), scaleSize(SIZE, j));

	//for (int LEVEL = 0; LEVEL < J; LEVEL++)
	//{
	//	for (int y = 0; y < scaleSize(SIZE, LEVEL); y++)
	//	{
	//		for (int x = 0; x < scaleSize(SIZE, LEVEL); x++)
	//			std::cout << result[LEVEL](x, y) << " ";
	//		std::cout << "\n";
	//	}
	//	std::cout << std::endl;

	//	for (int y = 0; y < scaleSize(SIZE, LEVEL); y++)
	//	{
	//		for (int x = 0; x < scaleSize(SIZE, LEVEL); x++)
	//			std::cout << stuff[LEVEL](x, y) << " ";
	//		std::cout << "\n";
	//	}
	//	std::cout << "\n\n";
	//}

	//return 0;
}
