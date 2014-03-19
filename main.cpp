#include "stdafx.h"

using namespace std;
using namespace Halide;
using namespace cv;

Var x("x"), y("y"), c("c");
const int WIDTH = 640, HEIGHT = 480, CHANNELS = 3;

template<typename F0>
double timing(F0 f, int iterations = 1)
{
	auto t0 = chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
		f();
	auto t1 = chrono::high_resolution_clock::now();
	double p = (double)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den;
	return (t1 - t0).count() * p / iterations;
}

// Prints timing in milliseconds
template<typename F0>
double printTiming(F0 f, string message = "", int iterations = 1)
{
	if (!message.empty())
		cout << message << flush;
	double t = timing(f, iterations);
	cout << setprecision(3) << 1000 * t << " ms" << endl;
	return t;
}

template<typename T>
Image<T> load2(string fileName = "images/in.png")
{
	Image<T> im;
	printTiming([&] { im = load<T>(fileName); }, "Loading " + fileName + "... ");
	return im;
}

template<typename T>
void save2(const Image<T>& im, string fileName = "images/out.png")
{
	printTiming([&] { save(im, fileName); }, "Saving to " + fileName + "... ");
}

template<typename T>
Func clipToEdges(const Image<T>& im)
{
	Func f;
	f(x, y, _) = im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1), _);
	return f;
}

Func clipToEdges(const ImageParam& im)
{
	Func f;
	f(x, y, _) = im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1), _);
	return f;
}

// Downsample with a 1 3 3 1 filter
Func downsample(Func f)
{
    Func downx, downy;

    downx(x, y, _) = (f(2*x-1, y, _) + 3.0f * (f(2*x, y, _) + f(2*x+1, y, _)) + f(2*x+2, y, _)) / 8.0f;
    downy(x, y, _) = (downx(x, 2*y-1, _) + 3.0f * (downx(x, 2*y, _) + downx(x, 2*y+1, _)) + downx(x, 2*y+2, _)) / 8.0f;

    return downy;
}

// Upsample using bilinear interpolation
Func upsample(Func f)
{
    Func upx, upy;

    upx(x, y, _) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, _) + 0.75f * f(x/2, y, _);
    upy(x, y, _) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), _) + 0.75f * upx(x, y/2, _);

    return upy;
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
template<int J>
array<Func, J> laplacianPyramid(array<Func, J> gPyramid)
{
	array<Func, J> lPyramid;
	lPyramid[J - 1](x, y, _) = gPyramid[J - 1](x, y, _);
	for (int j = J - 2; j >= 0; j--)
		lPyramid[j](x, y, _) = gPyramid[j](x, y, _) - upsample(gPyramid[j + 1])(x, y, _);
	return lPyramid;
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

int main_webcam()
{
	// Number of pyramid levels
	const int J = 8;
	// Number of entries in circular buffer.
	const int P = 5;
	const float alphaValues[J] = { 0, 0, 4, 7, 8, 9, 10, 10 };

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Cannot access webcam." << endl;
		return -1;
	}

	if (WIDTH != (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) || HEIGHT != (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT))
	{
		cerr << "Width and height are wrong." << endl;
		return -1;
	}

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
	array<Func, J> lPyramid = laplacianPyramid<J>(gPyramid);
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

	Mat frame;
	Mat outmat(HEIGHT, WIDTH, CV_8UC3, Scalar(0));
	Image<float> pyramidBuffer[P][J];
	Image<float> outputBuffer[P][J];
	for (int j = 0, w = WIDTH, h = HEIGHT; j < J; j++, w /= 2, h /= 2)
		for (int i = 0; i < P; i++)
			outputBuffer[i][j] = Image<float>(w, h);
	Image<float> outPyramid[J];
	for (int j = 0, w = WIDTH, h = HEIGHT; j < J; j++, w /= 2, h /= 2)
		outPyramid[j] = Image<float>(w, h);
	cap >> frame;	// Read one frame to initialize webcam.
	namedWindow("Out");
	double timeSum = 0;
	int frameCounter = -10;
	for (int i = 0;; i++, frameCounter++)
	{
		cap >> frame;
		double t = printTiming([&] {
			auto im = toImage_reorder(frame);
			input.set(im);
			for (int j = 0, w = WIDTH, h = HEIGHT; j < J; j++, w /= 2, h /= 2)
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
			Image<float> out = reconstruction.realize(WIDTH, HEIGHT, CHANNELS);
			toMat_reordered(out, outmat);
		}, "Processing... ");

		imshow("Out", outmat);
		if (waitKey(30) >= 0)
			break;

		if (frameCounter >= 0)
		{
			timeSum += t;
			cout << "(" << (frameCounter + 1) / timeSum << " FPS)" << endl;
		}
	}
	cout << "\nAverage FPS: " << frameCounter / timeSum << endl
		<< "Number of frames: " << frameCounter << endl;
	return 0;
}

int main()
{
	main_webcam();
}
