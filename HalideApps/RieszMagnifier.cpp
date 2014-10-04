#include "stdafx.h"
#include "RieszMagnifier.h"
#include "filter_util.h"
#include "Util.h"

using namespace Halide;

RieszMagnifier::RieszMagnifier(VideoApp app, int pyramidLevels, double freqCenter, double freqWidth)
	: app(app), pyramidLevels(pyramidLevels), freqCenter(freqCenter), freqWidth(freqWidth),
	input(ImageParam(Float(32), 3, "input")), bandAlphas({ 0.9375f, 1.875f, 3.75f, 7.5f, 15.0f, 30.0f, 30.0f }), output(Func("output"))
{
	Var x("x"), y("y"), c("c");
	Param<int> zero; zero.set(0);	// For buffers that aren't vectors of images.

	// Initialize pyramid buffers
	for (int j = 0; j < pyramidLevels; j++)
	{
		pyramidBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		r1Buffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		r2Buffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j), CIRCBUFFER_SIZE));
		phaseCBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		phaseSBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		lowpass1CBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		lowpass2CBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		lowpass1SBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
		lowpass2SBuffer.push_back(Image<float>(scaleSize(app.width(), j), scaleSize(app.height(), j)));
	}

	// Initialize temporal filter coefficients
	computeFilter();
	computePerBandAlpha();

	// Greyscale input
	Func grey("grey");
	grey(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

	// Gaussian pyramid
	std::vector<Func> gPyramid = makeFuncArray(pyramidLevels, "gPyramid");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j < pyramidLevels; j++)
	{
		gPyramid[j](x, y) = downsampleG5(clipToEdges(gPyramid[j - 1], scaleSize(app.width(), j - 1), scaleSize(app.height(), j - 1)))(x, y);
	}

	// Laplacian pyramid
	std::vector<Func> lPyramid = makeFuncArray(pyramidLevels, "lPyramid");
	lPyramid[pyramidLevels - 1](x, y) = gPyramid[pyramidLevels - 1](x, y);
	for (int j = pyramidLevels - 2; j >= 0; j--)
	{
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(clipToEdges(gPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}
	std::vector<Func> lPyramidCopy = copyPyramidToCircularBuffer(pyramidLevels, lPyramid, pyramidBuffer, pParam, "lPyramidCopy");

	// R1 pyramid
	std::vector<Func> r1Pyramid = makeFuncArray(pyramidLevels, "r1Pyramid");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramidCopy[j], scaleSize(app.width(), j), scaleSize(app.height(), j));
		r1Pyramid[j](x, y) = -0.6f * clamped(x - 1, y) + 0.6f * clamped(x + 1, y);
	}
	std::vector<Func> r1Copy = copyPyramidToCircularBuffer(pyramidLevels, r1Pyramid, r1Buffer, pParam, "r1Copy");

	// R2 pyramid
	std::vector<Func> r2Pyramid = makeFuncArray(pyramidLevels, "r2Pyramid");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramidCopy[j], scaleSize(app.width(), j), scaleSize(app.height(), j));
		r2Pyramid[j](x, y) = -0.6f * clamped(x, y - 1) + 0.6f * clamped(x, y + 1);
	}
	std::vector<Func> r2Copy = copyPyramidToCircularBuffer(pyramidLevels, r2Pyramid, r2Buffer, pParam, "r2Copy");

	// quaternionic phase difference as a tuple
	std::vector<Func> qPhaseDiff = makeFuncArray(pyramidLevels, "qPhaseDiff");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// q x q_prev*
		Expr productReal = lPyramidCopy[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			+ r1Copy[j](x, y) * r1Buffer[j](x, y, (pParam + 1) % 2)
			+ r2Copy[j](x, y) * r2Buffer[j](x, y, (pParam + 1) % 2);
		Expr productI = r1Copy[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			- r1Buffer[j](x, y, (pParam + 1) % 2) * lPyramidCopy[j](x, y);
		Expr productJ = r2Copy[j](x, y) * pyramidBuffer[j](x, y, (pParam + 1) % 2)
			- r2Buffer[j](x, y, (pParam + 1) % 2) * lPyramidCopy[j](x, y);

		Expr ijAmplitudeSq = productI * productI + productJ * productJ + 1e-6f;
		Expr amplitude = sqrt(ijAmplitudeSq + productReal * productReal);

		// normalizedProductReal = q x q_prev^-1 = q x q_prev* / ||q * q_prev||
		Expr normalizedProductReal = productReal / amplitude;
		Expr phi = acos(normalizedProductReal);
		Expr normalizedI = productI / sqrt(ijAmplitudeSq);
		Expr normalizedJ = productJ / sqrt(ijAmplitudeSq);

		qPhaseDiff[j](x, y) = Tuple(normalizedI * phi, normalizedJ * phi);
	}

	// Cumulative sums
	std::vector<Func> phaseC = makeFuncArray(pyramidLevels, "phaseC"),
		phaseS = makeFuncArray(pyramidLevels, "phaseS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		phaseC[j](x, y) = phaseCBuffer[j](x, y) + qPhaseDiff[j](x, y)[0];
		phaseS[j](x, y) = phaseSBuffer[j](x, y) + qPhaseDiff[j](x, y)[1];
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
		changeC[j](x, y) = (float)filterB[0] * phaseCBuffer[j](x, y) + lowpass1CBuffer[j](x, y);
		lowpass1C[j](x, y) = (float)filterB[1] * phaseCBuffer[j](x, y) + lowpass2CBuffer[j](x, y) - (float)filterA[1] * changeC[j](x, y);
		lowpass2C[j](x, y) = (float)filterB[2] * phaseCBuffer[j](x, y) - (float)filterA[2] * changeC[j](x, y);

		changeS[j](x, y) = (float)filterB[0] * phaseSBuffer[j](x, y) + lowpass1SBuffer[j](x, y);
		lowpass1S[j](x, y) = (float)filterB[1] * phaseSBuffer[j](x, y) + lowpass2SBuffer[j](x, y) - (float)filterA[1] * changeS[j](x, y);
		lowpass2S[j](x, y) = (float)filterB[2] * phaseSBuffer[j](x, y) - (float)filterA[2] * changeS[j](x, y);
	}
	std::vector<Func> lowpass1CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1C, lowpass1CBuffer, zero, "lowpass1CCopy");
	std::vector<Func> lowpass2CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2C, lowpass2CBuffer, zero, "lowpass2CCopy");
	std::vector<Func> lowpass1SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1S, lowpass1SBuffer, zero, "lowpass1SCopy");
	std::vector<Func> lowpass2SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2S, lowpass2SBuffer, zero, "lowpass2SCopy");

	std::vector<Func> changeCTuple = makeFuncArray(pyramidLevels, "changeCTuple"),
		changeSTuple = makeFuncArray(pyramidLevels, "changeSTuple"),
		changeC2 = makeFuncArray(pyramidLevels, "changeC2"),
		changeS2 = makeFuncArray(pyramidLevels, "changeS2");
	for (int j = 0; j < pyramidLevels; j++)
	{
		changeCTuple[j](x, y) = { phaseCCopy[j](x, y), changeC[j](x, y), lowpass1CCopy[j](x, y), lowpass2CCopy[j](x, y) };
		changeSTuple[j](x, y) = { phaseSCopy[j](x, y), changeS[j](x, y), lowpass1SCopy[j](x, y), lowpass2SCopy[j](x, y) };
		changeC2[j](x, y) = changeCTuple[j](x, y)[1];
		changeS2[j](x, y) = changeSTuple[j](x, y)[1];
	}

	std::vector<Func> amp = makeFuncArray(pyramidLevels, "amp"),
		changeCAmp = makeFuncArray(pyramidLevels, "changeCAmp"),
		changeCReg = makeFuncArray(pyramidLevels, "changeCReg"),
		changeSAmp = makeFuncArray(pyramidLevels, "changeSAmp"),
		changeSReg = makeFuncArray(pyramidLevels, "changeSReg"),
		ampReg = makeFuncArray(pyramidLevels, "ampReg"),
		magC = makeFuncArray(pyramidLevels, "magC"),
		magC2 = makeFuncArray(pyramidLevels, "magC2");
	std::vector<Func> outLPyramid = makeFuncArray(pyramidLevels, "outLPyramid");
	for (int j = 0; j < pyramidLevels; j++)
	{
		float sigma = 3;

		amp[j](x, y) = sqrt(lPyramid[j](x, y) * lPyramid[j](x, y)
			+ r1Pyramid[j](x, y) * r1Pyramid[j](x, y)
			+ r2Pyramid[j](x, y) * r2Pyramid[j](x, y)) + 1e-3f;

		ampReg[j](x, y) = gaussianBlur(clipToEdges(amp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma)(x, y);

		changeCAmp[j](x, y) = changeC2[j](x, y) * amp[j](x, y);
		changeCReg[j](x, y) = gaussianBlur(clipToEdges(changeCAmp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma)(x, y)
			/ ampReg[j](x, y);

		changeSAmp[j](x, y) = changeS2[j](x, y) * amp[j](x, y);
		changeSReg[j](x, y) = gaussianBlur(clipToEdges(changeSAmp[j], scaleSize(app.width(), j), scaleSize(app.height(), j)), sigma)(x, y)
			/ ampReg[j](x, y);

		magC[j](x, y) = hypot(changeCReg[j](x, y), changeSReg[j](x, y)) + 1e-3f;
		magC2[j](x, y) = 30.0f * magC[j](x, y);

		Expr pair = (r1Pyramid[j](x, y) * changeCReg[j](x, y) + r2Pyramid[j](x, y) * changeSReg[j](x, y)) / magC[j](x, y);
		outLPyramid[j](x, y) = lPyramid[j](x, y) * cos(magC2[j](x, y)) - pair * sin(magC2[j](x, y));
	}

	std::vector<Func> outGPyramid = makeFuncArray(pyramidLevels, "outGPyramid");
	outGPyramid[pyramidLevels - 1](x, y) = outLPyramid[pyramidLevels - 1](x, y);
	for (int j = pyramidLevels - 2; j >= 0; j--)
	{
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsample(clipToEdges(outGPyramid[j + 1], scaleSize(app.width(), j + 1), scaleSize(app.height(), j + 1)))(x, y);
	}

	output(x, y, c) = clamp(outGPyramid[0](x, y) * input(x, y, c) / (grey(x ,y) + 1e-6f), 0.0f, 1.0f);

	// Schedule
	output.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 4).vectorize(x, 4);
	for (int j = 0; j < pyramidLevels; j++)
	{
		outGPyramid[j].compute_root();
		outLPyramid[j].compute_root();
		magC[j].compute_root();
		amp[j].compute_root();
		ampReg[j].compute_root();
		changeCAmp[j].compute_root();
		changeCReg[j].compute_root();
		changeSAmp[j].compute_root();
		changeSReg[j].compute_root();
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
		changeC[j].compute_root();
		changeS[j].compute_root();
		phaseCCopy[j].compute_root();
		phaseSCopy[j].compute_root();
		phaseC[j].compute_root();
		phaseS[j].compute_root();
		qPhaseDiff[j].compute_root();
		r1Copy[j].compute_root();
		r2Copy[j].compute_root();
		r1Pyramid[j].compute_root();
		r2Pyramid[j].compute_root();
		lPyramidCopy[j].compute_root();
		lPyramid[j].compute_root();
		gPyramid[j].compute_root();
		gPyramid[j]
			.bound(x, 0, scaleSize(app.width(), j))
			.bound(y, 0, scaleSize(app.height(), j));

		if (j <= 4)
		{
			changeCTuple[j].vectorize(x, 4).parallel(y, 4);
			changeSTuple[j].vectorize(x, 4).parallel(y, 4);
			lPyramid[j].vectorize(x, 4).parallel(y, 4);
			gPyramid[j].vectorize(x, 4).parallel(y, 4);
		}
		else
		{
			lPyramid[j].parallel(y);
			gPyramid[j].parallel(y);
		}
	}

	// Compile JIT
	std::cout << "Compiling... ";
	output.compile_jit();
	std::cout << "done!" << std::endl;
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
	const double fps = 10.0;
	double lowCutoff = freqCenter - freqWidth / 2;
	double highCutoff = freqCenter + freqWidth / 2;

	// TODO: Should recompute the fps periodically, not assume fixed value
	filter_util::butterBP(1, { lowCutoff / (fps / 2.0), highCutoff / (fps / 2.0) }, filterA, filterB);
	filterA[1] /= filterA[0];
	filterA[2] /= filterA[0];
}

void RieszMagnifier::computePerBandAlpha()
{
	// TODO
}
