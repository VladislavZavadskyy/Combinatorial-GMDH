#include "utils.cuh"

Matrix computeY(Matrix& X, Matrix& coefs, bool addNoise)
{
	Matrix Y = X.dot(coefs);
	if (addNoise) {
		Matrix noise(Y.nrows(), Y.ncols(), normalInitializer<double>);
		if (Y._isCuda()) noise.cuda();
		Y = Y + noise;
	}
	return Y;
}

double computeCriterion(Matrix& X, Matrix& Y, Matrix& inferedCoefs)
{
	Matrix diff = X.dot(inferedCoefs) - Y;
	double numerator = (diff*diff)._reduce<double>(sum<double>());
	double denominator = (Y*Y)._reduce<double>(sum<double>());

	return numerator / denominator;
}

double computeUnbiasedness(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix &Y)
{
	Matrix infA = leastSquares(Xt, Yt);
	Matrix infB = leastSquares(Xval, Yval);

	Matrix diff = X.dot(infA) - X.dot(infB);
	Matrix diffSq = diff*diff;
	return (diff*diff)._reduce<double>(sum<double>()) / (Y*Y)._reduce<double>(sum<double>());
}

Matrix leastSquares(Matrix& X, Matrix& Y)
{
	Matrix X_t = std::move(X.T());
	Matrix xtdotx = std::move(X_t.dot(X));
	Matrix inv = std::move(xtdotx.inv());
	Matrix xtdoty = std::move(X_t.dot(Y));
	return inv.dot(xtdoty);
}

bool** generateMasks(size_t cols)
{
	const size_t masknum = pow(2, cols);
	bool** masks = (bool**)malloc(sizeof(bool*)*(masknum-1));
	for (int i = 1; i < masknum; ++i)
	{
		masks[i-1] = (bool*)malloc(sizeof(bool*)*cols);
		for (int j = 0; j < cols; ++j)
			masks[i-1][j] = (i >> j) & 1;
	}
	return masks;
}

void printMasks(bool** masks, size_t cols, bool print_sample)
{
	const size_t masknum = pow(2, cols)-1;
	printf("%llu masks in total (mask of zeros is ignored).\n"
		   "Brace yourself, for masks are comming: \n", masknum);
	Matrix X(1, cols, indexInitializer<double>);
	if (print_sample)X.cuda();
	for (int i = 0; i < masknum; ++i) {
		printMask(masks[i], cols);
		if (print_sample) X.filterCols(masks[i]).print();
		printf("\n");
	}
	printf("\n");
}

void printMask(bool* mask, size_t cols)
{
	for (int j = 0; j < cols; ++j)
		printf("%d ", mask[j]);
	printf("\n");
}

void deleteMasks(bool** masks, size_t cols)
{
	const size_t masknum = pow(2, cols)-1;
	for (int i = 0; i < masknum; ++i)
		free(masks[i]);
	free(masks);
}

using namespace std;

Matrix csv2mat(char * path)
{
	vector<vector<double>> vec = parseCSV(path);
	return vector2Matrix(vec);
}

Matrix vector2Matrix(vector<vector<double>> vector)
{
	size_t rows = vector.size();
	size_t cols = vector.front().size();
	Matrix M(rows, cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			M.seti(i, j, vector[i][j]);
	return M;
}

vector<vector<double>> parseCSV(char * path)
{
	string line;
	ifstream file(path);
	vector<vector<double>> parsed;
	if (file.is_open())
	{
		while (getline(file, line)) {
			vector<double> row = parseDoubleCSVString(line.c_str());
			if (row.size()!=0)
				parsed.push_back(row);
		}
		file.close();
	}
	return parsed;
}

vector<double> parseDoubleCSVString(const char* s)
{
	vector<double> parsed;
	vector<string> strings = split(s, ',');
	try {
		for (string el : strings)
			parsed.push_back(stod(el));
	} catch (invalid_argument){}
	return parsed;
}

vector<string> split(const char *str, char c)
{
	vector<string> result;

	do{
		const char *begin = str;
		while (*str != c && *str) str++;
		result.push_back(string(begin, str));
	} while (0 != *str++);

	return result;
}