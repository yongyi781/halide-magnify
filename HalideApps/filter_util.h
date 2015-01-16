namespace filter_util
{
	void zpk2tf(std::vector<std::complex<double>> zeros,
		std::vector<std::complex<double>> poles,
		double gain,
		std::vector<std::complex<double>>& a,
		std::vector<std::complex<double>>& b);
	void bilinear(std::vector<std::complex<double>>& b,
		std::vector<std::complex<double>>& a,
		double fs = 1.0);
	void lp2lp(std::vector<std::complex<double>>& b,
		std::vector<std::complex<double>>& a,
		double wo = 1.0);

	void compute_buttap(unsigned N,
		std::vector<std::complex<double>>& zeros,
		std::vector<std::complex<double>>& poles,
		double& gain);

	std::vector<std::complex<double> > poly(std::vector<std::complex<double>>);

	void normalize(std::vector<std::complex<double>>& b,
		std::vector<std::complex<double>>& a);

	unsigned comb(unsigned n,
		unsigned k);

	void butter(unsigned int N,
		std::vector<double> Wn,
		std::vector<double>& out_a,
		std::vector<double>& out_b,
		std::string btype = "lowpass",
		bool analog = false,
		std::string ftype = "butter");

	void butterBP(unsigned int N,
		std::vector<double> Wn,
		std::vector<double>& out_a,
		std::vector<double>& out_b);
	void computeFilter(double fps,
		double freqCenter,
		double freqWidth,
		std::vector<double>& filterA,
		std::vector<double>& filterB);
}
