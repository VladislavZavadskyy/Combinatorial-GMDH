#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#include "dependencies.cuh"
#include "utils.cuh"
#include "ops.cuh"

#define NORMAL_MEAN 0
#define NORMAL_STD 5
#define CEIL(a,b) (1 + ((a - 1) / b))

class incompatibleDimsException : public std::exception
{
private:
	const char * message;
public:
	incompatibleDimsException(char* message) : message(message) {};

	const char* what() const throw() override
	{
		if (message != nullptr)
			return message;
		return "Matrices have incompatible dimensions";
	}
};

class placementException : public std::exception
{
	const char* what() const throw() override
	{
		return "Matrices must be placed on the same device";
	}
};

extern unsigned long long counter;

/**
* \brief A wrapper class for 2D arrays
* \tparam T - datatype
*/
template<class T>
T zeroInitializer(int i, int j) {
	return static_cast<T>(0);
}

template<class T>
T onesInitializer(int i, int j) {
	return static_cast<T>(1);
}

template<class T>
T indexInitializer(int i, int j) {
	return static_cast<T>(i + 1e-2*j);
}

template<class T>
T eyeInitializer(int i, int j) {
	return static_cast<T>(i == j);
}

template<class T>
T zebraInitializer(int i, int j) {
	if (i % 2 == 0)
		return static_cast<T>(0);
	return static_cast<T>(1);
}

template<class T>
T normalInitializer(int i, int j) {
	static std::default_random_engine generator;
	static std::normal_distribution<double> distribution(NORMAL_MEAN, NORMAL_STD);
	return distribution(generator);
}

class Matrix
{
	size_t rows;
	size_t cols;
	size_t stride;
	bool moved;
	bool initialized=false;
	double* values;
	bool isCuda;
	cudaStream_t stream;
	int idx(int row, int col) const;
	void copy(const Matrix& another);
	void move(Matrix& another);

public:
	Matrix(size_t nrows, size_t ncols, double* values, bool isCuda, size_t stride_, bool moved_, bool initialized_);
	Matrix(size_t nrows, size_t ncols, double(*initializer)(int, int));
	Matrix(size_t nrows, size_t ncols);
	Matrix(Matrix &another);
	Matrix(Matrix&& another) noexcept;
	~Matrix();

	void cuda();
	void cpu();
	
	Matrix operator+(Matrix& A);
	Matrix operator-(Matrix& B);
	Matrix operator*(Matrix& B);
	Matrix operator/(Matrix& B);
	Matrix operator=(const Matrix&);
	Matrix operator=(Matrix&& another) noexcept;

	Matrix identity() const;
	Matrix dot(Matrix &another) const;
	Matrix T() const;
	Matrix inv();

	Matrix colSum() const;
	Matrix rowSum() const;
	
	Matrix insertRow(int index, double* values);
	Matrix insertCol(int index, double* values);
	Matrix insertRow(int index, double (*initializer)(int, int));
	Matrix insertCol(int index, double (*initializer)(int, int));
	Matrix filterCols(bool* mask);

	Matrix sub(size_t row, size_t col, size_t dimx, size_t dimy) const;

	void print() const;
	double geti(int row, int col) const;
	void seti(int row, int col, double value);
	bool equal(Matrix &another, double eps = 1e-4) const;

	size_t getStride() const { return stride; }
	size_t nrows() const { return rows; }
	size_t ncols() const { return cols; }
	bool _isCuda() const { return isCuda; }

	template <class T, class F>
	T _reduce(F Op);

	template<class F>
	friend Matrix elementwiseOp(Matrix& A, Matrix& B);

	friend void CUDAPipeline(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix& Y,
		int *d_remap, int newcols, cudaStream_t * streams, cublasHandle_t *cublasHandles,
		cusolverDnHandle_t *cuSolverHandles, double *d_num_t, double *d_num_val, double** pointers);

	friend void cudaPrecomputeDenominators(Matrix& Yt, Matrix Yval, double &denom_t, double &denom_val, cudaStream_t *streams);
};

template<class F>
Matrix elementwiseOp(Matrix& A, Matrix& B)
{
	if (A.rows != B.rows || A.cols != B.cols)
		throw incompatibleDimsException(nullptr);
	Matrix C(A.rows, A.cols);
	if (!A.isCuda && !B.isCuda) {
		F op = F();
		for (int i = 0; i < A.rows; ++i)
			for (int j = 0; j < A.cols; ++j)
				C.seti(i, j, op(A.geti(i, j), B.geti(i, j)));
		return C;
	}
	if (A.isCuda && B.isCuda) {
		C.cuda();
		dim3 * dims = getOneDims(A.rows*A.cols);
		elementwise <<< dims[0], dims[1], 0, C.stream >>> (A.values, B.values, A.rows*A.cols, C.values, F());
		cud(cudaStreamSynchronize(C.stream));
		return C;
	}
	throw placementException();
}

template<class T, class F>
T Matrix::_reduce(F Op)
{
	T value;
	if (!isCuda) {
		value = Op.identity;
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				value = Op(value, geti(i, j));
	}
	else {
		T *d_value;
		size_t size = rows*cols;
		cud(cudaMalloc((void **)&d_value, sizeof(T)));
		reduce(this->values, size, Op, &d_value, stream);
		cud(cudaMemcpyAsync(&value, d_value, sizeof(T), cudaMemcpyDeviceToHost, stream));
		cud(cudaFree(d_value));
	}
	return value;
}

__host__ void matPrint(size_t rows, size_t cols, size_t stride, double* values);
void getCudaEyeAsynch(double *d_eye, size_t dim, cudaStream_t stream);

#endif