#ifndef MATOP_CUH_
#define MATOP_CUH_

#include "dependencies.cuh"
#include "utils.cuh"
#include "matrix.cuh"

#define CEIL(a,b) (1 + ((a - 1) / b))
/**
 * \brief (CUDA) Transposes a matrix
 * \param odata - pointer to output 2D array (on device)
 * \param idata - pointer to input 2D array (on device)
 */
__global__ void deviceTranspose(double *odata, const double *idata, int width, int height);
/**
 * \brief (CUDA) Fills and identity 2D array placed on device
 * \param devMatrix - pointer to the array placed on device
 * \param height - number of rows in the array
 * \param width - number of cols in the array
 */
__global__ void eye(double *values, int rows, int cols);
/**
 * \brief Calculates dimensions for CUDA kernel call
 * \param height - 2D array height
 * \param width - 2D array width
 * \return 
 */
__global__ void deviceFilter(const size_t cols, const size_t rows, double* unfiltered, double* filtered, int * remap);

__host__ dim3* getMaxGrid(size_t rows, size_t cols);
__host__ dim3* getTransposeGridDims(size_t height, size_t width, size_t tiledim);
__host__ dim3* getDotGridDims(size_t Arows, size_t Bcols, size_t tiledim);
__host__ dim3* getOneDims(size_t size);

__global__ void deviceDot(double* A, double* B, double* C, size_t Acols, size_t Bcols, size_t Arows);
template<class T, class F>
__global__ void elementwise(double* A, double* B, size_t size, T *C, F Op);
__global__ void deviceCopy(double *A, size_t lda, double *B, size_t ldb, size_t row, size_t col, size_t dimy, size_t dimx);

template<typename T, typename F>
__global__ void reduceKernel(T* input, int size, T* output, F Op);

template<typename T, typename F> void reduce(T* input, const int size, F Op, T **d_result, cudaStream_t stream);

template<typename T, typename F>
__global__ void reduceKernel(T* input, int size, T* output, F Op)
{
	extern __shared__ __align__(sizeof(T)) unsigned char smem[];
	T *sData = reinterpret_cast<T*>(smem);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	
	sData[threadIdx.x] = Op.identity;
	if (gid >= size) return;
	sData[threadIdx.x] = input[gid];
	__syncthreads();

	double div_val = (double(blockDim.x) / 2);
	int offset = (div_val) < 1 ? 0 : ceil(div_val);

	for (; offset > 0; ) {
		if (threadIdx.x < offset) {
			#ifdef DEBUG
			printf("off: %d (%d:%d && %d:%d) -> %d:%d\n", offset, threadIdx.x, sData[threadIdx.x], threadIdx.x + offset,
				sData[threadIdx.x + offset], threadIdx.x, Op(sData[threadIdx.x], sData[threadIdx.x + offset])); 
			#endif
			sData[threadIdx.x] = Op(sData[threadIdx.x], sData[threadIdx.x + offset]);
			sData[threadIdx.x + offset] = Op.identity;
		} else break;
		div_val = (double(offset) / 2);
		offset = (div_val) < 1 ? 0 : ceil(div_val);
		__syncthreads();
	}

	if (threadIdx.x == 0) output[blockIdx.x] = sData[0];
}

template<typename T, typename F>
void reduce(T* input, const int size, F Op, T **d_result, cudaStream_t stream)
{
	dim3 * dims = getOneDims(size);

	T *d_mid;
	cud(cudaMalloc((void **)&d_mid, sizeof(T)*dims[0].x));

	reduceKernel <<< dims[0], dims[1], sizeof(T) * dims[1].x, stream >>>(input, size, d_mid, Op);
	dims = getOneDims(dims[0].x);
	cud(cudaStreamSynchronize(stream));
	reduceKernel <<<  dims[0], dims[1], sizeof(T) * dims[1].x, stream >>>(d_mid, dims[0].x, *d_result, Op);
	cud(cudaStreamSynchronize(stream));
	cud(cudaFree(d_mid));
}

void invPipelineCUDA(size_t dim, cudaStream_t *streams, cusolverDnHandle_t handle, double * d_values, double * &d_result);

template<class T, class F>
__global__ void elementwise(double* A, double* B, size_t size, T* C, F Op)
{
	size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= size) return;
	C[tid] = Op(A[tid], B[tid]);
}

struct AndFn {
	__host__ __device__ bool operator()(bool a, bool b) { return a && b; }
	bool identity = true;
};

struct equality {
	double eps = 1e-3;
	__host__ __device__ bool operator()(double a, double b) { return abs(a - b) < eps; }
	bool identity = true;
};

template<class T>
struct sum {
	__host__ __device__  T operator()(T a, T b) { return a+b; }
	T identity = 0;
};

template<class T>
struct diff {
	__host__ __device__  T operator()(T a, T b) { return a - b; }
	T identity = 0;
};

template<class T>
struct product {
	__host__ __device__  T operator()(T a, T b) { return a * b; }
	T identity = 1;
};

template<class T>
struct division {
	__host__ __device__  T operator()(T a, T b) { return a / b; }
	T identity = 1;
};

template<class T>
struct absoluteDiff {
	__host__ __device__  T operator()(T a, T b) { return abs(a - b); }
	T identity = 0;
};

using namespace std;
class Matrix;
void CUDAPipeline(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix& Y,
	int *d_remap, int newcols, cudaStream_t * streams, cublasHandle_t *cublasHandles,
	cusolverDnHandle_t *cuSolverHandles, double *d_num_t, double *d_num_val, double** pointers);
void createCUDAWorkspace(cudaStream_t * &streams, cusolverDnHandle_t * &cusdolverHandles, cublasHandle_t * &cublasHandles);
void destroyCUDAWorkspace(cudaStream_t * &streams, cusolverDnHandle_t * &cusdolverHandles, cublasHandle_t * &cublasHandles);
void cudaPrecomputeDenominators(Matrix& Yt, Matrix Yval, double &denom_t, double &denom_val, cudaStream_t *streams);
double** allocateDeviceMemory(size_t rows, size_t cols);
void deallocateDeviceMemory(void** allocations);

//allocated in order of appearence here
//int*
//0:d_remap, 1:d_piv, 2:d_info;
//double*
//3:d_Xt_filtered, 4:d_Xval_filtered, 5:d_X_filtered;
//6:d_XTdotX_t, 7:d_XTdotX_val;
//8:d_XTdotY_t, 9:d_XTdotY_val;
//10:d_XTdotX_inv_t, 11:d_XTdotX_inv_val;
//12:d_coefs_t, 13:d_coefs_val;
//14:d_Y_hat_t, 15:d_Y_hat_val;
//16:diff_t, 17:diff_val;
//18:d_diff_t_sq, 19:d_diff_val_sq;
//20:d_Yt_sq, 21:d_Yval_sq;
//double**
//22:d_matrices, 23:d_inverted; 

const struct {
	int d_remap, d_piv, d_info,
		d_Xt_filtered, d_Xval_filtered,
		d_XTdotX_t, d_XTdotX_val,
		d_XTdotY_t, d_XTdotY_val,
		d_XTdotX_inv_t, d_XTdotX_inv_val,
		d_coefs_t, d_coefs_val,
		d_Y_hat_t, d_Y_hat_val,
		diff_t, diff_val,
		d_diff_t_sq, d_diff_val_sq,
		d_Yt_sq, d_Yval_sq,
		d_matrices, d_inverted;
} s = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 
		9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22 };
#endif