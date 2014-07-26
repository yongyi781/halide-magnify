#include "stdafx.h"
#include "EulerianMagnifier.h"
#include "Util.h"

#define TILE 1

using namespace Halide;

EulerianMagnifier::EulerianMagnifier(VideoApp app, int pyramidLevels) : app(app), pyramidLevels(pyramidLevels),
	input(ImageParam(Float(32), 3)), alphaValues({ 0, 0, 2, 5, 10, 10, 10, 10, 10 }), output(Func("output"))
{
	Var x("x"), y("y"), c("c");

	// Initialize pyramid buffers
	for (int j = 0; j < pyramidLevels; j++)
	{
		pyramidBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		temporalOutBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
	}

	Func grey("grey"); grey(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

	// Gaussian pyramid
	std::vector<Func> gPyramid(pyramidLevels);
	gPyramid[0] = Func("gPyramid0");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j < pyramidLevels; j++)
	{
		gPyramid[j] = Func("gPyramid" + std::to_string(j));
		gPyramid[j](x, y) = downsample(clipToEdges(gPyramid[j - 1], scaleSize(app.width(), j - 1), scaleSize(app.height(), j - 1)))(x, y);
	}

	// Laplacian pyramid
	std::vector<Func> lPyramid(pyramidLevels);
	lPyramid[pyramidLevels - 1] = Func("lPyramid" + std::to_string(pyramidLevels - 1));
	lPyramid[pyramidLevels - 1](x, y) = gPyramid[pyramidLevels - 1](x, y);
	for (int j = pyramidLevels - 2; j >= 0; j--)
	{
		lPyramid[j] = Func("lPyramid" + std::to_string(j));
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(clipToEdges(gPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	// Copy to pyramid buffer
	std::vector<Func> lPyramidWithCopy(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
	{
		Param<buffer_t*> copyToParam;
		copyToParam.set(pyramidBuffer[j].raw_buffer());
		lPyramidWithCopy[j] = Func("lPyramidWithCopy" + std::to_string(j));
		lPyramidWithCopy[j].define_extern("copyFloat32", { pParam, copyToParam, lPyramid[j] }, Float(32), 2);
	}

	std::vector<Func> temporalProcess(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
	{
		temporalProcess[j] = Func("temporalProcess" + std::to_string(j));
		temporalProcess[j](x, y) =
			1.1430f * temporalOutBuffer[j](x, y, (pParam - 2 + 5) % 5)
			- 0.4128f * temporalOutBuffer[j](x, y, (pParam - 4 + 5) % 5)
			+ 0.6389f * lPyramidWithCopy[j](x, y)
			- 1.2779f * pyramidBuffer[j](x, y, (pParam - 2 + 5) % 5)
			+ 0.6389f * pyramidBuffer[j](x, y, (pParam - 4 + 5) % 5);
	}

	std::vector<Func> temporalProcessWithCopy(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
	{
		Param<buffer_t*> copyToParam;
		copyToParam.set(temporalOutBuffer[j].raw_buffer());
		temporalProcessWithCopy[j] = Func("temporalProcessWithCopy" + std::to_string(j));
		temporalProcessWithCopy[j].define_extern("copyFloat32", { pParam, copyToParam, temporalProcess[j] }, Float(32), 2);
	}

	std::vector<Func> outLPyramid(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
	{
		outLPyramid[j] = Func("outLPyramid" + std::to_string(j));
		outLPyramid[j](x, y) = lPyramid[j](x, y) + (alphaValues[j] == 0.0f ? 0.0f : alphaValues[j] * temporalProcessWithCopy[j](x, y));
	}

	std::vector<Func> outGPyramid(pyramidLevels);
	outGPyramid[pyramidLevels - 1] = Func("outGPyramid" + std::to_string(pyramidLevels - 1));
	outGPyramid[pyramidLevels - 1](x, y) = outLPyramid[pyramidLevels - 1](x, y);
	for (int j = pyramidLevels - 2; j >= 0; j--)
	{
		outGPyramid[j] = Func("outGPyramid" + std::to_string(j));
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsample(clipToEdges(outGPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	output(x, y, c) = clamp(outGPyramid[0](x, y) * input(x, y, c) / (0.01f + grey(x, y)), 0.0f, 1.0f);

	// Schedule
	Var xi("xi"), yi("yi");

	output.reorder(c, x, y).bound(c, 0, app.channels()).unroll(c).vectorize(x, 4).parallel(y, 4);
#if TILE
	output.tile(x, y, xi, yi, app.width() / 8, app.height() / 8);
#endif

	for (int j = 0; j < pyramidLevels; j++)
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

	std::cout << "Compiling... ";
	output.compile_jit();
	std::cout << "done!" << std::endl;
}

void EulerianMagnifier::process(Image<float> frame, Image<float> out)
{
	pParam.set(frameCounter % CIRCBUFFER_SIZE);
	input.set(frame);
	output.realize(out);

	frameCounter++;
}
