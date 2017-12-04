#include "matrix.cuh"
#define LAPACK
#define CUDAMALLOC0
Matrix::Matrix(size_t nrows, size_t ncols)
{
	rows = nrows;
	cols = ncols;
	isCuda = false;
	stream = 0;
	stride = rows;
	moved = false;
#ifdef CUDAMALLOC
	cud(cudaMallocHost((void**)&values, rows * cols * sizeof(double)));
#else
	values = static_cast<double*>(malloc(rows * cols * sizeof(double)));
#endif
	initialized = false;
}

Matrix::Matrix(size_t nrows, size_t ncols, double(*initializer)(int, int))
{
	rows = nrows;
	cols = ncols;
	isCuda = false;
	stream = 0;
	stride = rows;
	moved = false;
#ifdef CUDAMALLOC
	cud(cudaMallocHost((void**)&values, rows * cols * sizeof(double)));
#else
	values = static_cast<double*>(malloc(rows * cols * sizeof(double)));
#endif
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			values[idx(i,j)] = initializer(i,j);
	initialized = true;
}

Matrix::Matrix(size_t nrows, size_t ncols, double* Values, bool cuda, 
	size_t stride_, bool moved_, bool initialized_)
{
	isCuda = cuda;
	rows = nrows;
	cols = ncols;
	values = Values;
	stride = stride_;
	moved = moved_;
	initialized = initialized_;
	if (isCuda) {
		stream = cudaStream_t();
		cud(cudaStreamCreate(&stream));
	}
}

Matrix::Matrix(Matrix &another)
{
	copy(another);
}

Matrix::Matrix(Matrix&& another) noexcept
{
	move(another);
}

Matrix::~Matrix()
{
	if (!moved) {
		if (isCuda) {
			cud(cudaStreamSynchronize(stream));
			cud(cudaFree(values));
			cud(cudaStreamDestroy(stream));
		}
#ifdef CUDAMALLOC
		else cudaFreeHost(values);
#else
		else free(values);
#endif
	}
}


void Matrix::move(Matrix& another)
{
	if (initialized) this->~Matrix();
	if (another.isCuda) {
		stream = cudaStream_t();
		cud(cudaStreamCreate(&stream));
		cud(cudaStreamDestroy(another.stream));
	} else stream = 0;
	isCuda = std::move(another.isCuda);
	rows = std::move(another.rows);
	cols = std::move(another.cols);
	stride = std::move(another.stride);
	values = std::move(another.values);
	initialized = std::move(another.initialized);
	moved = false;
	another.moved = true;
}

void Matrix::copy(const Matrix& another)
{
	if (initialized) this->~Matrix();
	if (another.isCuda) {
		stream = cudaStream_t();
		cud(cudaStreamCreate(&stream));
	} else stream = 0;
	isCuda = another.isCuda;
	rows = another.rows;
	cols = another.cols;
	stride = another.stride;
	moved = another.moved;
	initialized = another.initialized;
	if (isCuda) {
		cud(cudaMalloc((void **)&values, sizeof(double)*rows*cols));
		if (initialized) {
			counter++;
			cud(cudaMemcpyAsync(values, another.values, rows*cols * sizeof(double),
				cudaMemcpyDeviceToDevice, stream));
			cud(cudaStreamSynchronize(stream));
		}
	} else {
		if (initialized) {
#ifdef CUDAMALLOC
			cud(cudaMallocHost((void**)&values, rows * cols * sizeof(double)));
			cud(cudaMemcpy(values, another.values, rows*cols * sizeof(double), cudaMemcpyHostToHost));
#else
			values = static_cast<double*>(malloc(rows * cols * sizeof(double)));
			memcpy(values, another.values, rows*cols * sizeof(double));
#endif
		}
	}
}

Matrix Matrix::operator=(const Matrix& another)
{
	copy(another);
	return *this;
}

Matrix Matrix::operator=(Matrix&& another) noexcept
{
	move(another);
	return *this;
}

Matrix Matrix::operator+(Matrix& B)
{
	return elementwiseOp<sum<double>>(*this, B);
}

Matrix Matrix::operator-(Matrix& B)
{
	return elementwiseOp<diff<double>>(*this, B);
}

Matrix Matrix::operator*(Matrix& B)
{
	return elementwiseOp<product<double>>(*this, B);
}

Matrix Matrix::operator/(Matrix& B)
{
	return elementwiseOp<division<double>>(*this, B);
}

void Matrix::cuda()
{
	if (!isCuda) {
		double *dvalues;
		stream = cudaStream_t();
		cud(cudaStreamCreate(&stream));
		cud(cudaMalloc((void **)&dvalues, rows * cols * sizeof(double)));
		if (initialized) {
			cud(cudaMemcpyAsync(dvalues, values, rows * cols * sizeof(double),
				cudaMemcpyHostToDevice, stream));
			cud(cudaStreamSynchronize(stream));
		}
#ifdef CUDAMALLOC
		cud(cudaFreeHost(values));
#else
		free(values);
#endif
		values = dvalues;
		isCuda = true;
	} else {
		printf("The matrix is already in GPU memory space.");
	}
}

void Matrix::cpu()
{
	if (isCuda) {
		double* hvalues;
#ifdef CUDAMALLOC
		cud(cudaMallocHost((void**)&hvalues, rows * cols * sizeof(double)));
#else
		hvalues = static_cast<double*>(malloc(rows * cols * sizeof(double)));
#endif
		if (initialized) {
			cud(cudaMemcpyAsync(hvalues, values, rows * cols * sizeof(double),
				cudaMemcpyDeviceToHost, stream));
			cud(cudaStreamSynchronize(stream));
		}
		cud(cudaFree(values));
		values = hvalues;
		isCuda = false;
		cud(cudaStreamDestroy(stream));
	} else {
		printf("The matrix is already in CPU memory space.");
	}
}

int Matrix::idx(int row, int col) const
{
	return col*stride + row;
}

double Matrix::geti(int row, int col) const
{
	if (!initialized) throw _exception();
	if (isCuda) {
		double * value = (double*)malloc(sizeof(double));
		cud(cudaMemcpyAsync(value, &values[idx(row, col)], sizeof(double), cudaMemcpyDeviceToHost, stream));
		cud(cudaStreamSynchronize(stream));
		return *value;
	}
	return values[idx(row, col)];
}

void Matrix::seti(int row, int col, double value)
{
	if (isCuda)
	{
		cud(cudaMemcpyAsync(&values[idx(row, col)], &value, sizeof(double), cudaMemcpyHostToDevice, stream));
		cud(cudaStreamSynchronize(stream));
	} else
		values[idx(row, col)] = value;
	initialized = true;
}

bool Matrix::equal(Matrix &another, double eps) const
{
	if (cols != another.cols || rows != another.rows)
		return false;
	if (!isCuda && !another.isCuda) {
		for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			if (abs(geti(i, j) - another.geti(i, j)) > eps) {
				#ifdef DEBUG
				printf("%.4f!=%.4f; on (%d,%d)\n", geti(i, j), another.geti(i, j), i, j);
				#endif
				return false;
			}
		return true;
	}
	if(isCuda && another.isCuda){
		size_t size = rows*cols;
		dim3 * dims = getOneDims(size);
		bool * elwiseEquality;
		cud(cudaMalloc((void **)&elwiseEquality, sizeof(bool)*size));
		equality op;
		op.eps = eps;
		elementwise <<< dims[0], dims[1], 0, stream >>> (values, another.values, size, elwiseEquality, op);
		cud(cudaStreamSynchronize(stream));
		#ifdef DEBUG
		bool *hwise = new bool[size];
		cud(cudaMemcpyAsync(hwise, elwiseEquality, size * sizeof(bool), cudaMemcpyDeviceToHost, stream));
		cud(cudaStreamSynchronize(stream));
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j)
				printf("%d ", hwise[rows*j + i]);
			printf("\n");
		}
		delete[] hwise;
		#endif
		bool *d_result, result;
		cud(cudaMalloc((void **)d_result, sizeof(bool)));
		reduce(elwiseEquality, size, AndFn(), &d_result, stream);
		cud(cudaMemcpyAsync(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost, stream));
		cud(cudaStreamSynchronize(stream));
		cud(cudaFree(elwiseEquality));
		cud(cudaFree(d_result));
		return result;
	}
	throw placementException();
}

Matrix Matrix::T() const
{
	if (!initialized) throw _exception();
	Matrix transposed(cols, rows);

#ifndef LAPACK
	if (isCuda) {
		dim3* dims = getTransposeGridDims(rows, cols, 4);
		cudaTranspose << < dims[0], dims[1], dims[2].x, stream >> > (transposed.values, values, cols, rows);
	} else {
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					transposed.seti(j, i, geti(i, j));
	}	
#else
	if (isCuda) {
		transposed.cuda();
		cublasHandle_t handle;
		cublas(cublasCreate(&handle));
		cublas(cublasSetStream(handle, stream));
		const double one = 1.0, zero = 0.0;
		cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, &one, values, rows,
			&zero, values, rows, transposed.values, cols));
		cud(cudaStreamSynchronize(stream));
		cublas(cublasDestroy(handle))
	} else {
		mkl_domatcopy('C', 'T', rows, cols, 1.0, values, stride, transposed.values, transposed.stride);
	}
#endif
	transposed.initialized = true;
	return transposed;
}

__global__ void cuPrint(size_t rows, size_t cols, size_t stride, double * values)
{
	int tid = threadIdx.y*gridDim.x + threadIdx.x;
	if (tid == 0) {
		printf("\nMatrix on GPU with dimensions (%llu,%llu)\n",rows,cols);
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
				printf("%.3e ", values[j*stride + i]);
			printf("\n");
		}
	}
}

__host__ Matrix Matrix::sub(size_t row, size_t col, size_t dimx, size_t dimy) const
{
	if (!initialized) throw _exception();
	Matrix sub(dimy, dimx);
	sub.initialized = initialized;
	if (!isCuda)
		for (int i = 0; i < dimy; ++i)
			for (int j = 0; j < dimx; ++j)
				sub.seti(i, j, geti(row + i, col + j));
	else {
		sub.cuda();
		if (initialized) {
			dim3 * dims = getMaxGrid(dimy, dimx);
			deviceCopy <<< dims[0], dims[1], 0, sub.stream >>> (values, stride, sub.values, sub.stride, row, col, dimy, dimx);
			cud(cudaStreamSynchronize(sub.stream));
		}
	}
	return sub;
}

void getCudaEyeAsynch(double *d_eye, size_t dim, cudaStream_t stream)
{
	dim3 * dims = getMaxGrid(dim, dim);
	eye <<< dims[0], dims[1], 0, stream >>> (d_eye, dim, dim);
}

//Warning: inplace LU-factorization
#define THRESHOLD 0
Matrix Matrix::inv() 
{
	if (!initialized) throw _exception();
	if (rows != cols)
		throw incompatibleDimsException("Matrix must be square for inv() operation");
	if (!isCuda || rows<THRESHOLD) {
		size_t buffSize = std::max(size_t(1), std::min(rows, cols));
		int *ipiv = (int *)malloc(sizeof(int)*buffSize);
		Matrix Copy(*this);
		if (isCuda) Copy.cpu();
		LAPACKE_dgetrf(LAPACK_COL_MAJOR, rows, cols, Copy.values, stride, ipiv);
		LAPACKE_dgetri(LAPACK_COL_MAJOR, cols, Copy.values, stride, ipiv);
		if (isCuda) Copy.cuda();
		return Copy;
	}
	cusolverDnHandle_t handle;
	cusolve(cusolverDnCreate(&handle));
	cusolve(cusolverDnSetStream(handle, stream));
	Matrix Eye(rows, rows); Eye.cuda();
	cudaStream_t *streams = new cudaStream_t[2]{ stream, Eye.stream };
	invPipelineCUDA(rows, streams, handle, values, std::ref(Eye.values));
	Eye.initialized = true;
	cusolve(cusolverDnDestroy(handle));
	delete[] streams;
	return Eye;
}

void Matrix::print() const
{
	if (!initialized) throw _exception();
	if (isCuda) {
		cuPrint << < 1, 1, 0, stream >> > (rows, cols, stride, values);
		cud(cudaStreamSynchronize(stream));
	}
	else
		matPrint(rows, cols, stride, values);
}

void matPrint(size_t rows, size_t cols, size_t stride, double* values)
{
	printf("\nMatrix on CPU with dimensions (%llu,%llu)\n", rows, cols);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
			printf("%.3e ", values[j*stride +i]);
		printf("\n");
	}
}

Matrix Matrix::dot(Matrix &another) const
{
	if (!initialized) throw _exception();
	if (cols != another.rows) {
		std::cout << "Matrices are incompatible" << std::endl;
		throw incompatibleDimsException(nullptr);
	}
	Matrix product(rows, another.cols);
	const double one = 1, zero = 0;
#ifndef LAPACK
		if (isCuda && another.isCuda) {
			product.cuda();
			dim3 * dims = getDotGridDims(rows, another.cols, std::min(rows, cols));
			deviceDot << < dims[0], dims[1], dims[2].x, stream >> > (values, another.values, product.values, cols, another.cols, rows);
		} else if (isCuda && another.isCuda) {
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < another.cols; ++j)
					for (int k = 0; k < cols; ++k)
						product.values[product.idx(i, j)] += geti(i, k) * another.geti(k, j);
		} else throw placementException();
#else
	if (isCuda && another.isCuda) {
		product.cuda();
		cublasHandle_t handle;
		cublas(cublasCreate(&handle));
		cublas(cublasSetStream(handle, stream));
		cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, another.cols, cols, &one,
			values, stride, another.values, another.stride, &zero, product.values, product.stride));
		cud(cudaStreamSynchronize(stream));
		cublas(cublasDestroy(handle));
	} else if (!isCuda && !another.isCuda) {
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, another.cols, cols, 1.0, values, stride,
			another.values, another.stride, 0.0, product.values, product.stride);
	} else throw placementException();
#endif
	product.initialized = true;
	return product;
}

Matrix Matrix::identity() const
{
	return Matrix(rows, cols, values, isCuda, stride, true, initialized);
}

Matrix Matrix::rowSum() const
{
	if (!initialized) throw _exception();
	Matrix Ones(rows, 1, onesInitializer<double>);
	if (isCuda) Ones.cuda();
	return T().dot(Ones).T();
}

Matrix Matrix::colSum() const
{
	if (!initialized) throw _exception();
	Matrix Ones(cols, 1, onesInitializer<double>);
	if (isCuda) Ones.cuda();
	return dot(Ones);
}

Matrix Matrix::insertRow(int index, double *values)
{
	if (!initialized) throw _exception();
	const bool wasCuda = isCuda;
	if (isCuda) cpu();
	Matrix newOne(rows + 1, cols);
	if (index == -1) index = rows;
	int shift = 0;
	for (int i = 0; i < rows+1; ++i) {
		if (i == index) {
			for (int j = 0; j < cols; ++j)
				newOne.seti(i, j, values[j]);
			shift = -1;
		} else
			for (int j = 0; j < cols; ++j)
				newOne.seti(i, j, geti(i + shift, j));
	}
	if (wasCuda) newOne.cuda();
	newOne.initialized = true;
	return newOne;
}

Matrix Matrix::insertCol(int index, double *vs)
{
	if (!initialized) throw _exception();
	const bool wasCuda = isCuda;
	if (isCuda) cpu();
	Matrix newOne(rows, cols+1);
	if (index == -1) index = cols;
	int shift = 0;
	for (int j = 0; j < cols+1; ++j) {
		if (j == index) {
			for (int i = 0; i < rows; ++i)
				newOne.seti(i, j, vs[i]);
			shift = -1;
		} else
			for (int i = 0; i < rows; ++i)
				newOne.seti(i, j, geti(i + shift, j));
	}
	if (wasCuda) newOne.cuda();
	newOne.initialized = true;
	return newOne;
}

Matrix Matrix::insertRow(int index, double(*initializer)(int, int))
{
	if (!initialized) throw _exception();
	if (index == -1) index = rows;
	double *vs = (double*)malloc(sizeof(double)*cols);
	for (int i = 0; i < cols; ++i)
		vs[i] = initializer(index, i);
	return insertCol(index, vs);
}

Matrix Matrix::insertCol(int index, double(*initializer)(int, int))
{
	if (!initialized) throw _exception();
	if (index == -1) index = cols;
	double *vs = (double*)malloc(sizeof(double)*rows);
	for (int i = 0; i < rows; ++i)
		vs[i] = initializer(i, index);
	return insertCol(index, vs);
}

Matrix Matrix::filterCols(bool * mask)
{
	if (!initialized) throw _exception();
	size_t newcols = 0;
	int *remap = (int*)malloc(sizeof(int)*cols);
	for (int i = 0; i < cols; ++i)
		if (mask[i]) {
			remap[newcols] = i;
			newcols++;
		}
	Matrix Filtered(rows, newcols);
	if (!isCuda) {
		for (int i = 0; i < newcols; ++i)
			for (int j = 0; j < rows; ++j)
				Filtered.seti(j,i, geti(j,remap[i]));
	} else {
		Filtered.cuda();
		dim3 * dims = getMaxGrid(rows, newcols);
		int * d_remap;
		cud(cudaMalloc((void**)&d_remap, sizeof(int)*cols));
		cud(cudaMemcpyAsync(d_remap, remap, sizeof(int)*cols, cudaMemcpyHostToDevice, stream));
		deviceFilter<<<dims[0],dims[1],0,stream>>>(newcols, rows, values, Filtered.values, d_remap);
		cud(cudaStreamSynchronize(stream));
		cud(cudaFree(d_remap));
	}
	Filtered.initialized = true;
	free(remap);
	return Filtered;
}