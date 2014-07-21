// HalideApps.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "WebcamApp.h"
#include "NamedWindow.h"

using namespace Halide;

#define TRACE 0
#define TILE 1

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

// Downsample with a 1 2 1 filter
template<typename F>
Func downsample(F f)
{
	Func downx("downx"), downy("downy");

	downx(x, y, _) = (f(2 * x - 1, y, _) + 2.0f * f(2 * x, y, _) + f(2 * x + 1, y, _)) / 4.0f;
	downy(x, y, _) = (downx(x, 2 * y - 1, _) + 2.0f * downx(x, 2 * y, _) + downx(x, 2 * y + 1, _)) / 4.0f;

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

#pragma endregion

// Number of pyramid levels
const int PYRAMID_LEVELS = 5;
// Size of circular buffer
const int CIRCBUFFER_SIZE = 5;

std::array<Image<float>, PYRAMID_LEVELS> pyramidBuffer;
std::array<Image<float>, PYRAMID_LEVELS> temporalOutBuffer;

// Extern function to copy data to an external buffer.
extern "C" __declspec(dllexport) int copyFloat32(int p, buffer_t* copyTo, buffer_t* in, buffer_t* out)
{
	if (in->host == nullptr || out->host == nullptr)
	{
		if (in->host == nullptr)
		{
			for (int i = 0; i < 2; i++)
			{
				in->min[i] = out->min[i];
				in->extent[i] = out->extent[i];
			}
		}
	}
	else
	{
#if TRACE
		printf("[Copy] p: %d out: [%d, %d] x [%d, %d], in: [%d, %d] x [%d, %d]\n", p,
			out->min[0], out->min[0] + out->extent[0] - 1, out->min[1], out->min[1] + out->extent[1] - 1,
			in->min[0], in->min[0] + in->extent[0] - 1, in->min[1], in->min[1] + in->extent[1] - 1);
#endif
		float* src = (float*)in->host;
		float* dst = (float*)out->host;
		float* dstCopy = (float*)copyTo->host + p * copyTo->stride[2];
		for (int y = out->min[1]; y < out->min[1] + out->extent[1]; y++)
		{
			float* srcLine = src + (y - in->min[1]) * in->stride[1];
			float* dstLine = dst + (y - out->min[1]) * out->stride[1];
			float* copyLine = dstCopy + y * copyTo->stride[1];
			memcpy(dstLine, srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
			memcpy(copyLine + out->min[0], srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
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
// TODO:
// - Try bigger image (upsample webcam input)
// - Tile everything with compute_at x
int main_v3()
{
	const float alphaValues[PYRAMID_LEVELS] = { 0, 0, 2, 5, 10 };
	Param<int> pParam;

	WebcamApp app(2);
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

	Func grey("grey"); grey(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

	// Gaussian pyramid
	Func gPyramid[PYRAMID_LEVELS];
	gPyramid[0] = Func("gPyramid0");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j < PYRAMID_LEVELS; j++)
	{
		gPyramid[j] = Func("gPyramid" + std::to_string(j));
		gPyramid[j](x, y) = downsample(clipToEdges(gPyramid[j - 1], scaleSize(app.width(), j - 1), scaleSize(app.height(), j - 1)))(x, y);
	}

	// Laplacian pyramid
	Func lPyramid[PYRAMID_LEVELS];
	lPyramid[PYRAMID_LEVELS - 1] = Func("lPyramid" + std::to_string(PYRAMID_LEVELS - 1));
	lPyramid[PYRAMID_LEVELS - 1](x, y) = gPyramid[PYRAMID_LEVELS - 1](x, y);
	for (int j = PYRAMID_LEVELS - 2; j >= 0; j--)
	{
		lPyramid[j] = Func("lPyramid" + std::to_string(j));
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(clipToEdges(gPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	// Copy to pyramid buffer
	Func lPyramidWithCopy[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		Param<buffer_t*> copyToParam;
		copyToParam.set(pyramidBuffer[j].raw_buffer());
		lPyramidWithCopy[j] = Func("lPyramidWithCopy" + std::to_string(j));
		lPyramidWithCopy[j].define_extern("copyFloat32", { pParam, copyToParam, lPyramid[j] }, Float(32), 2);
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
		Param<buffer_t*> copyToParam;
		copyToParam.set(temporalOutBuffer[j].raw_buffer());
		temporalProcessWithCopy[j] = Func("temporalProcessWithCopy" + std::to_string(j));
		temporalProcessWithCopy[j].define_extern("copyFloat32", { pParam, copyToParam, temporalProcess[j] }, Float(32), 2);
	}

	Func outLPyramid[PYRAMID_LEVELS];
	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
		outLPyramid[j] = Func("outLPyramid" + std::to_string(j));
		outLPyramid[j](x, y) = lPyramid[j](x, y) + (alphaValues[j] == 0.0f ? 0.0f : alphaValues[j] * temporalProcessWithCopy[j](x, y));
	}

	Func outGPyramid[PYRAMID_LEVELS];
	outGPyramid[PYRAMID_LEVELS - 1] = Func("outGPyramid" + std::to_string(PYRAMID_LEVELS - 1));
	outGPyramid[PYRAMID_LEVELS - 1](x, y) = outLPyramid[PYRAMID_LEVELS - 1](x, y);
	for (int j = PYRAMID_LEVELS - 2; j >= 0; j--)
	{
		outGPyramid[j] = Func("outGPyramid" + std::to_string(j));
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsample(clipToEdges(outGPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	Func output("output");
	output(x, y, c) = clamp(outGPyramid[0](x, y) * input(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Schedule
	Var xi("xi"), yi("yi");

	output.reorder(c, x, y).bound(c, 0, app.channels()).unroll(c).vectorize(x, 4).parallel(y, 4);
#if TILE
	output.tile(x, y, xi, yi, app.width() / 8, app.height() / 8);
#endif

	for (int j = 0; j < PYRAMID_LEVELS; j++)
	{
#if TILE
		outGPyramid[j].compute_at(output, x);
		temporalProcessWithCopy[j].compute_at(output, x);
		temporalProcess[j].compute_at(output, x);
		lPyramidWithCopy[j].compute_at(output, x);
		lPyramid[j].compute_at(output, x);
		gPyramid[j].compute_at(output, x);
#else
		outGPyramid[j].compute_root();
		temporalProcessWithCopy[j].compute_root();
		temporalProcess[j].compute_root();
		lPyramidWithCopy[j].compute_root();
		lPyramid[j].compute_root();
		gPyramid[j].compute_root();
		outGPyramid[j]
			.bound(x, 0, scaleSize(app.width(), j))
			.bound(y, 0, scaleSize(app.height(), j));
		temporalProcess[j]
			.bound(x, 0, scaleSize(app.width(), j))
			.bound(y, 0, scaleSize(app.height(), j));
		lPyramid[j]
			.bound(x, 0, scaleSize(app.width(), j))
			.bound(y, 0, scaleSize(app.height(), j));
		gPyramid[j]
			.bound(x, 0, scaleSize(app.width(), j))
			.bound(y, 0, scaleSize(app.height(), j));
#endif

		if (j <= 4)
		{
			outGPyramid[j].vectorize(x, 4).parallel(y, 4);
			lPyramid[j].vectorize(x, 4).parallel(y, 4);
			gPyramid[j].vectorize(x, 4).parallel(y, 4);
		}
		else
		{
			outGPyramid[j].parallel(y);
			lPyramid[j].parallel(y);
			gPyramid[j].parallel(y);
		}
	}

	// Compile
	std::cout << "Compiling...";
	output.compile_jit();
	std::cout << "\nDone compiling!\n";

	NamedWindow window("Results");
	Image<float> frame;
	Image<float> out(app.width(), app.height(), app.channels());
	double timeSum = 0;
	int frameCounter = -10;
	for (int i = 0;; i++, frameCounter++)
	{
		frame = app.readFrame();
		if (frame.dimensions() == 0)
			break;
		int p = i % CIRCBUFFER_SIZE;
		pParam.set(p);
		input.set(frame);

		if (i < CIRCBUFFER_SIZE - 1)
		{
			output.realize(app.width(), app.height(), app.channels());
		}
		else
		{
			double t = currentTime();
			// --- timing ---
			output.realize(out);
			// --- end timing ---
			double diff = currentTime() - t;
			window.showImage(out);
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

int main_bug()
{
	const int WIDTH = 1280, HEIGHT = 960;
	Image<float> in = lambda(x, y, cast<float>(x + y)).realize(WIDTH, HEIGHT);

	Func gPyramid1("gPyramid1"), gPyramid2("gPyramid2"), lPyramid1("lPyramid1"), lPyramid2("lPyramid2");
	gPyramid1(x, y) = downsample(clipToEdges(in))(x, y);
	gPyramid2(x, y) = downsample(clipToEdges(in))(x, y);
	lPyramid1(x, y) = in(x, y) - upsample(clipToEdges(gPyramid1, WIDTH / 2, HEIGHT / 2))(x, y);
	lPyramid2(x, y) = in(x, y) - upsample(clipToEdges(gPyramid2, WIDTH / 2, HEIGHT / 2))(x, y);

	Var xi("xi"), yi("yi");
	lPyramid2.tile(x, y, xi, yi, WIDTH / 2, HEIGHT / 2);

	Image<float> out1 = lPyramid1.realize(WIDTH / 2, HEIGHT / 2);
	Image<float> out2 = lPyramid2.realize(WIDTH / 2, HEIGHT / 2);

	for (int y = 0; y < HEIGHT / 2; y++)
		for (int x = 0; x < WIDTH / 2; x++)
			if (out1(x, y) != out2(x, y))
				std::cerr << "Mismatch at (" << x << ", " << y << ")" << std::endl;

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
