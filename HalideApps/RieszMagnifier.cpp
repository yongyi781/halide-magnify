#include "stdafx.h"
#include "RieszMagnifier.h"
#include "filter_util.h"
#include "Util.h"

#define TILE 0

using namespace Halide;

const float TINY = 1e-24f;

RieszMagnifier::RieszMagnifier(VideoApp app, int pyramidLevels, double freqCenter, double freqWidth, float alpha)
	: app(app), pyramidLevels(pyramidLevels), freqCenter(freqCenter), freqWidth(freqWidth), alpha(alpha),
	input(ImageParam(Float(32), 3, "input")), bandAlphas(std::vector<float>(pyramidLevels)), output(Func("output"))
{
	Var x("x"), y("y"), c("c");
	Param<int> zero{ 0 };	// For buffers that aren't vectors of images, so pass in 0 to 'p' argument.

	// Initialize pyramid buffers
	for (int j = 0; j < pyramidLevels; j++)
	{
		pyramidBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));

		phaseCBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		phaseSBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		lowpass1CBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		lowpass2CBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		lowpass1SBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		lowpass2SBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
	}

	// Initialize temporal filter coefficients
	computeFilter();
	computeBandAlphas();

	// Greyscale input
	Func grey("grey");
	grey(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

	// CbCr - 0.5
	Func cb("cb"), cr("cr");
	cb(x, y) = -0.168736f * input(x, y, 0) - 0.331264f * input(x, y, 1) + 0.5f * input(x, y, 2);
	cr(x, y) = 0.5f * input(x, y, 0) - 0.418688f * input(x, y, 1) - 0.081312f * input(x, y, 2);

	// Gaussian pyramid
	std::vector<Func> gPyramid = makeFuncArray(pyramidLevels + 1, "gPyramid");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j < gPyramid.size(); j++)
	{
		gPyramid[j](x, y) = downsample5(clipToEdges(gPyramid[j - 1], scaleSize(app.width(), j - 1), scaleSize(app.height(), j - 1)))(x, y);
	}

	// Laplacian pyramid
	std::vector<Func> lPyramid = makeFuncArray(pyramidLevels + 1, "lPyramid");
	lPyramid[pyramidLevels](x, y) = gPyramid[pyramidLevels](x, y);
	for (int j = pyramidLevels - 1; j >= 0; j--)
	{
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(clipToEdges(gPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}
	std::vector<Func> lPyramidCopy = copyPyramidToCircularBuffer(pyramidLevels, lPyramid, pyramidBuffer, pParam, "lPyramidCopy");

	std::vector<Func> clampedPyramidBuffer = makeFuncArray(pyramidLevels, "clampedPyramidBuffer");
	for (int j = 0; j < pyramidLevels; j++)
	{
		clampedPyramidBuffer[j](x, y, _) = clipToEdges(pyramidBuffer[j])(x, y, _);
	}

	// R1 pyramid
	std::vector<Func> r1Pyramid = makeFuncArray(pyramidLevels, "r1Pyramid"),
		r1Prev = makeFuncArray(pyramidLevels, "r1Prev");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramid[j], scaleSize(app.width(), j), scaleSize(app.height(), j));
		r1Pyramid[j](x, y) = -0.6f * clamped(x - 1, y) + 0.6f * clamped(x + 1, y);
		r1Prev[j](x, y) = -0.6f * clampedPyramidBuffer[j](x - 1, y, (pParam + 1) % 2) + 0.6f * clampedPyramidBuffer[j](x + 1, y, (pParam + 1) % 2);
	}

	// R2 pyramid
	std::vector<Func> r2Pyramid = makeFuncArray(pyramidLevels, "r2Pyramid"),
		r2Prev = makeFuncArray(pyramidLevels, "r2Prev");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramid[j], scaleSize(app.width(), j), scaleSize(app.height(), j));
		r2Pyramid[j](x, y) = -0.6f * clamped(x, y - 1) + 0.6f * clamped(x, y + 1);
		r2Prev[j](x, y) = -0.6f * clampedPyramidBuffer[j](x, y - 1, (pParam + 1) % 2) + 0.6f * clampedPyramidBuffer[j](x, y + 1, (pParam + 1) % 2);
	}

	// quaternionic phase difference as a tuple
	std::vector<Func>
		productReal = makeFuncArray(pyramidLevels, "productReal"),
		productI = makeFuncArray(pyramidLevels, "productI"),
		productJ = makeFuncArray(pyramidLevels, "productJ"),
		ijAmplitude = makeFuncArray(pyramidLevels, "ijAmplitude"),
		amplitude = makeFuncArray(pyramidLevels, "amplitude"),
		phi = makeFuncArray(pyramidLevels, "phi"),
		qPhaseDiffC = makeFuncArray(pyramidLevels, "qPhaseDiffC"),
		qPhaseDiffS = makeFuncArray(pyramidLevels, "qPhaseDiffS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// q x q_prev*
		productReal[j](x, y) = lPyramidCopy[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			+ r1Pyramid[j](x, y) * r1Prev[j](x, y)
			+ r2Pyramid[j](x, y) * r2Prev[j](x, y);
		productI[j](x, y) = r1Pyramid[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			- r1Prev[j](x, y) * lPyramid[j](x, y);
		productJ[j](x, y) = r2Pyramid[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			- r2Prev[j](x, y) * lPyramid[j](x, y);

		ijAmplitude[j](x, y) = hypot(productI[j](x, y), productJ[j](x, y)) + TINY;
		amplitude[j](x, y) = hypot(ijAmplitude[j](x, y), productReal[j](x, y)) + TINY;

		// cos(phi) = q x q_prev^-1 = q x q_prev* / ||q * q_prev||
		phi[j](x, y) = acos(productReal[j](x, y) / amplitude[j](x, y)) / ijAmplitude[j](x, y);

		qPhaseDiffC[j](x, y) = productI[j](x, y) * phi[j](x, y);
		qPhaseDiffS[j](x, y) = productJ[j](x, y) * phi[j](x, y);
	}

	// Cumulative sums
	std::vector<Func> phaseC = makeFuncArray(pyramidLevels, "phaseC"),
		phaseS = makeFuncArray(pyramidLevels, "phaseS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		phaseC[j](x, y) = qPhaseDiffC[j](x, y) + phaseCBuffer[j](x, y);
		phaseS[j](x, y) = qPhaseDiffS[j](x, y) + phaseSBuffer[j](x, y);
	}

	std::vector<Func> phaseCCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseC, phaseCBuffer, zero, "phaseCCopy");
	std::vector<Func> phaseSCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseS, phaseSBuffer, zero, "phaseSCopy");

	std::vector<Func> changeC = makeFuncArray(pyramidLevels, "changeC"),
		lowpass1C = makeFuncArray(pyramidLevels, "lowpass1C"),
		lowpass2C = makeFuncArray(pyramidLevels, "lowpass2C"),
		changeS = makeFuncArray(pyramidLevels, "changeS"),
		lowpass1S = makeFuncArray(pyramidLevels, "lowpass1S"),
		lowpass2S = makeFuncArray(pyramidLevels, "lowpass2S");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// Linear filter. Order of evaluation here is important.
		changeC[j](x, y) = (float)filterB[0] * phaseCCopy[j](x, y) + lowpass1CBuffer[j](x, y, (pParam + 1) % 2);
		lowpass1C[j](x, y) = (float)filterB[1] * phaseCBuffer[j](x, y) + lowpass2CBuffer[j](x, y, (pParam + 1) % 2) - (float)filterA[1] * changeC[j](x, y);
		lowpass2C[j](x, y) = (float)filterB[2] * phaseCBuffer[j](x, y) - (float)filterA[2] * changeC[j](x, y);

		changeS[j](x, y) = (float)filterB[0] * phaseSCopy[j](x, y) + lowpass1SBuffer[j](x, y, (pParam + 1) % 2);
		lowpass1S[j](x, y) = (float)filterB[1] * phaseSBuffer[j](x, y) + lowpass2SBuffer[j](x, y, (pParam + 1) % 2) - (float)filterA[1] * changeS[j](x, y);
		lowpass2S[j](x, y) = (float)filterB[2] * phaseSBuffer[j](x, y) - (float)filterA[2] * changeS[j](x, y);
	}
	std::vector<Func> lowpass1CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1C, lowpass1CBuffer, pParam, "lowpass1CCopy");
	std::vector<Func> lowpass2CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2C, lowpass2CBuffer, pParam, "lowpass2CCopy");
	std::vector<Func> lowpass1SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1S, lowpass1SBuffer, pParam, "lowpass1SCopy");
	std::vector<Func> lowpass2SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2S, lowpass2SBuffer, pParam, "lowpass2SCopy");

	std::vector<Func> changeCTuple = makeFuncArray(pyramidLevels, "changeCTuple"),
		changeSTuple = makeFuncArray(pyramidLevels, "changeSTuple"),
		changeC2 = makeFuncArray(pyramidLevels, "changeC2"),
		changeS2 = makeFuncArray(pyramidLevels, "changeS2");
	for (int j = 0; j < pyramidLevels; j++)
	{
		changeCTuple[j](x, y) = { changeC[j](x, y), lowpass1CCopy[j](x, y), lowpass2CCopy[j](x, y) };
		changeSTuple[j](x, y) = { changeS[j](x, y), lowpass1SCopy[j](x, y), lowpass2SCopy[j](x, y) };
		changeC2[j](x, y) = changeCTuple[j](x, y)[0];
		changeS2[j](x, y) = changeSTuple[j](x, y)[0];
	}

	std::vector<Func> amp = makeFuncArray(pyramidLevels, "amp"),
		changeCAmp = makeFuncArray(pyramidLevels, "changeCAmp"),
		changeCRegX = makeFuncArray(pyramidLevels, "changeCRegX"),
		changeCReg = makeFuncArray(pyramidLevels, "changeCReg"),
		changeSAmp = makeFuncArray(pyramidLevels, "changeSAmp"),
		changeSRegX = makeFuncArray(pyramidLevels, "changeSRegX"),
		changeSReg = makeFuncArray(pyramidLevels, "changeSReg"),
		ampRegX = makeFuncArray(pyramidLevels, "ampRegX"),
		ampReg = makeFuncArray(pyramidLevels, "ampReg"),
		magC = makeFuncArray(pyramidLevels, "magC"),
		magC2 = makeFuncArray(pyramidLevels, "magC2"),
		outLPyramid = makeFuncArray(pyramidLevels, "outLPyramid");
	for (int j = 0; j < pyramidLevels; j++)
	{
		float sigma = 3;

		amp[j](x, y) = sqrt(lPyramid[j](x, y) * lPyramid[j](x, y)
			+ r1Pyramid[j](x, y) * r1Pyramid[j](x, y)
			+ r2Pyramid[j](x, y) * r2Pyramid[j](x, y)) + TINY;

		std::tie(ampRegX[j], ampReg[j]) = gaussianBlur(clipToEdges(amp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma, x, y);

		changeCAmp[j](x, y) = changeC2[j](x, y) * amp[j](x, y);
		Func tempC;
		std::tie(changeCRegX[j], tempC) = gaussianBlur(clipToEdges(changeCAmp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma, x, y);
		changeCReg[j](x, y) = tempC(x, y) / ampReg[j](x, y);

		changeSAmp[j](x, y) = changeS2[j](x, y) * amp[j](x, y);
		Func tempS;
		std::tie(changeSRegX[j], tempS) = gaussianBlur(clipToEdges(changeSAmp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma, x, y);
		changeSReg[j](x, y) = tempS(x, y) / ampReg[j](x, y);

		magC[j](x, y) = hypot(changeCReg[j](x, y), changeSReg[j](x, y)) + TINY;

		Expr pair = (r1Pyramid[j](x, y) * changeCReg[j](x, y) + r2Pyramid[j](x, y) * changeSReg[j](x, y)) / magC[j](x, y);
		outLPyramid[j](x, y) = bandAlphas[j] == 0.0f ? lPyramid[j](x, y) : lPyramid[j](x, y) * cos(bandAlphas[j] * magC[j](x, y)) - pair * sin(bandAlphas[j] * magC[j](x, y));
	}

	std::vector<Func> outGPyramid = makeFuncArray(pyramidLevels + 1, "outGPyramid");
	outGPyramid[pyramidLevels](x, y) = lPyramid[pyramidLevels](x, y);
	for (int j = pyramidLevels - 1; j >= 0; j--)
	{
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsample(clipToEdges(outGPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	// YCrCb -> RGB
	output(x, y, c) = clamp(select(
		c == 0, outGPyramid[0](x, y) + 1.402f * cr(x, y),
		c == 1, outGPyramid[0](x, y) - 0.34414f * cb(x, y) - 0.71414f * cr(x, y),
		outGPyramid[0](x, y) + 1.772f * cb(x, y)), 0.0f, 1.0f);

	//output(x, y, c) = outLPyramid[0](x, y);

	// Schedule
	output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
#ifndef PROFILE
	output.parallel(y, 4).vectorize(x, 4);
#endif

	for (int j = 0; j <= pyramidLevels; j++)
	{
		lPyramid[j].compute_root();
		gPyramid[j].compute_root();

#ifndef PROFILE
		if (j <= 4)
		{
			lPyramid[j].parallel(y, 4).vectorize(x, 4);
			gPyramid[j].parallel(y, 4).vectorize(x, 4);
		}
		else
		{
			lPyramid[j].parallel(y);
			gPyramid[j].parallel(y);
		}
#endif
	}

	Var xi, yi;
	for (int j = 0; j < pyramidLevels; j++)
	{
		outGPyramid[j].compute_root();
		outLPyramid[j].compute_root();

		ampRegX[j].compute_root();
		changeCRegX[j].compute_root();
		changeSRegX[j].compute_root();

		changeCTuple[j].compute_root();
		changeSTuple[j].compute_root();

		lowpass1CCopy[j].compute_root();
		lowpass2CCopy[j].compute_root();
		lowpass1SCopy[j].compute_root();
		lowpass2SCopy[j].compute_root();
		lowpass1C[j].compute_root();
		lowpass2C[j].compute_root();
		lowpass1S[j].compute_root();
		lowpass2S[j].compute_root();

		phaseCCopy[j].compute_root();
		phaseSCopy[j].compute_root();
		phaseC[j].compute_root();
		phaseS[j].compute_root();
		phi[j].compute_root();

		r1Pyramid[j].compute_root();
		r2Pyramid[j].compute_root();

		lPyramidCopy[j].compute_root();

#ifndef PROFILE
		if (j <= 4)
		{
			outGPyramid[j].parallel(y, 4).vectorize(x, 4);
			outLPyramid[j].parallel(y, 4).vectorize(x, 4);

			ampRegX[j].parallel(y, 4).vectorize(x, 4);
			changeCRegX[j].parallel(y, 4).vectorize(x, 4);
			changeSRegX[j].parallel(y, 4).vectorize(x, 4);

			changeCTuple[j].parallel(y, 4).vectorize(x, 4);
			changeSTuple[j].parallel(y, 4).vectorize(x, 4);

			lowpass1C[j].parallel(y, 4).vectorize(x, 4);
			lowpass2C[j].parallel(y, 4).vectorize(x, 4);
			lowpass1S[j].parallel(y, 4).vectorize(x, 4);
			lowpass2S[j].parallel(y, 4).vectorize(x, 4);

			phaseC[j].parallel(y, 4).vectorize(x, 4);
			phaseS[j].parallel(y, 4).vectorize(x, 4);
			phi[j].parallel(y, 4).vectorize(x, 4);

			r1Pyramid[j].parallel(y, 4).vectorize(x, 4);
			r2Pyramid[j].parallel(y, 4).vectorize(x, 4);
		}
		else
		{
			outGPyramid[j].parallel(y);
		}
#endif
	}

	// Compile JIT
	std::cout << "Compiling... ";
	double t = currentTime();
	output.compile_jit();
	std::cout << "done! Elapsed: " << (currentTime() - t) / 1000 << " ms" << std::endl;
}

void RieszMagnifier::process(const Halide::Image<float>& frame, const Image<float>& out)
{
	pParam.set(frameCounter % CIRCBUFFER_SIZE);
	input.set(frame);
	output.realize(out);

	frameCounter++;
}

void RieszMagnifier::computeFilter()
{
	double fps = app.fps();
	if (fps == 0.0)
		fps = 20.0;
	double lowCutoff = freqCenter - freqWidth / 2;
	double highCutoff = freqCenter + freqWidth / 2;

	// TODO: Should recompute the fps periodically, not assume fixed value
	filter_util::butterBP(1, { lowCutoff / (fps / 2.0), highCutoff / (fps / 2.0) }, filterA, filterB);
	filterA[1] /= filterA[0];
	filterA[2] /= filterA[0];
}

void RieszMagnifier::computeBandAlphas()
{
	for (int j = 0; j < pyramidLevels; j++)
	{
		bandAlphas[j] = j < 1 ? alpha / 2 : alpha;
	}
}
