#include "stdafx.h"
#include "RieszMagnifier.h"
#include "Util.h"

using namespace Halide;

RieszMagnifier::RieszMagnifier(VideoApp app, int pyramidLevels) : app(app), pyramidLevels(pyramidLevels),
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
	output(x, y, c) = 0.5f;
}

void RieszMagnifier::process(Image<float> frame, Image<float> out)
{
	pParam.set(frameCounter % CIRCBUFFER_SIZE);
	input.set(frame);
	output.realize(out);

	frameCounter++;
}
