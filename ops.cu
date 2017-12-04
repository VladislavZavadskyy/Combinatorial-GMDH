#include "ops.cuh"

#define DEBUG0
#define CEIL(a,b) (1 + ((a - 1) / b))

using namespace std;
__host__ dim3* getTransposeGridDims(size_t height, size_t width, size_t tiledim)
{
	dim3 dimBlock(min(min(tiledim, size_t(32)), width),
		min(min(tiledim, size_t(32)), height));
	dim3 dimGrid(max(CEIL(width, dimBlock.x), size_t(1)),
				 max(CEIL(height, dimBlock.y), size_t(1)));
	dim3 shMem(dimBlock.x*(dimBlock.y+1)*sizeof(double));
	return new dim3[3]{dimGrid, dimBlock, shMem};
}

__host__ dim3* getDotGridDims(size_t Arows, size_t Bcols, size_t tiledim)
{
	tiledim = min(tiledim, size_t(32));
	dim3 dimBlock(tiledim, tiledim);
	dim3 dimGrid(CEIL(Bcols, tiledim), CEIL(Arows, tiledim));
	dim3 shMem(2*tiledim*(tiledim+1)*sizeof(double));
	return new dim3[3]{ dimGrid, dimBlock, shMem};
}

__host__ dim3* getMaxGrid(size_t rows, size_t cols)
{
	size_t tiledim = min(max(rows,cols), size_t(32));
	const dim3 dimBlock(tiledim, tiledim);
	const dim3 dimGrid(CEIL(cols, tiledim), CEIL(rows, tiledim));
	return new dim3[2]{ dimGrid, dimBlock };
}

__host__ dim3* getOneDims(size_t size)
{
	int dim = min(size, size_t(1024));
	dim = (dim % 2 == 0) ? dim: dim + 1;
	dim3 dimBlock(dim);
	dim3 dimGrid(CEIL(size,dimBlock.x));
	dim3 shMem(dimBlock.x * sizeof(double));
	return new dim3[3]{ dimGrid, dimBlock, shMem };
}

__global__ void deviceFilter(const size_t cols, const size_t rows, double* unfiltered, double* filtered, int * remap)
{
	const size_t tx = blockDim.x*blockIdx.x + threadIdx.x;
	const size_t ty = blockDim.y*blockIdx.y + threadIdx.y;
	if (tx >= cols || ty >= rows) return;
	filtered[tx*rows + ty] = unfiltered[remap[tx]*rows+ty];
}

__global__ void deviceTranspose(double *odata, const double *idata, int width, int height)
{
	extern __shared__ double tile[];

	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int index_in = xIndex*height + yIndex;
	int index_out = yIndex*width + xIndex;
	if (index_out >= width*height) return;

	#ifdef DEBUG
	if (index_in == 0)
		printf("Grid: (%d,%d); Block: (%d,%d)\n", gridDim.x, 
			gridDim.y, blockDim.x, blockDim.y);
	__syncthreads();
	#endif
	int ix = threadIdx.y*blockDim.x + threadIdx.x;

	tile[ix] = idata[index_in];
	__syncthreads();
	odata[index_out] = tile[ix];
}

__global__ void eye(double *values, int rows, int cols) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (y < rows && x < cols)
		values[x*rows+y] = (x == y) ? 1.0 : 0.0;
}

__global__ void deviceDot(double* A, double* B, double* C, size_t Acols, size_t Bcols, size_t Arows)
{
	size_t tiledim = blockDim.x;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	double* Csub = &C[Bcols*tiledim*blockRow + tiledim*blockCol];
	double Cvalue = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m <= (Acols / tiledim); ++m) {
		double* Asub = &A[Acols*tiledim*blockRow + tiledim*m];
		double* Bsub = &B[Bcols*tiledim*m + tiledim*blockCol];

		extern __shared__ double shmem[];
		double *Bs = (double*)&shmem;
		double *As = (double*)&shmem[tiledim * (tiledim+1)];

		As[row*tiledim+col] = Asub[row*Acols + col];
		Bs[row*tiledim+col] = Bsub[row*Bcols + col];
		__syncthreads();

		for (int e = 0; e < tiledim; ++e)
			Cvalue += As[row*tiledim+e] * Bs[e*tiledim+col];
		__syncthreads();
	}
	
	if (col + blockDim.x*blockIdx.x < Bcols && row + blockDim.y*blockIdx.y < Arows)
		Csub[row*Bcols + col] = Cvalue;
}

__global__ void deviceCopy(double *A, size_t lda, double *B, size_t ldb, size_t row, size_t col, size_t dimy, size_t dimx)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dimx || y >= dimy) return;

	B[x*ldb+y] = A[(col+x)*lda + y+row];
}

//First sream should be assigned to handle, second stream should be free
void invPipelineCUDA(size_t dim, cudaStream_t *streams, cusolverDnHandle_t handle, double * d_values, double * &d_result)
{
	int *d_Ipiv, *d_info;
	int  lwork, info;
	double *d_work;

	cusolve(cusolverDnDgetrf_bufferSize(handle, dim, dim, d_values, dim, &lwork));
	getCudaEyeAsynch(d_result, dim, streams[1]);

	cud(cudaMalloc((void**)&d_Ipiv, sizeof(int) * dim));
	cud(cudaMalloc((void**)&d_info, sizeof(int)));
	cud(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

	cusolve(cusolverDnDgetrf(handle, dim, dim, d_values, dim, d_work, d_Ipiv, d_info));
	cud(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
	cud(cudaStreamSynchronize(streams[0]));

	if (info < 0){
		printf("Error #%d after factorization, aborting.\n", info);
		goto CLEAR;
	}

	cud(cudaStreamSynchronize(streams[1]));
	cusolve(cusolverDnDgetrs(handle, CUBLAS_OP_N, dim, dim, d_values, dim, d_Ipiv, d_result, dim, d_info));
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
	cud(cudaStreamSynchronize(streams[0]));
	if (info < 0) printf("Error #%d after factorization, aborting.\n", info);
CLEAR:
	cud(cudaFree(d_Ipiv));
	cud(cudaFree(d_info));
	cud(cudaFree(d_work));
}

void CUDAPipeline(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix& Y,
	int *d_remap, int newcols, cudaStream_t * streams, cublasHandle_t *cublasHandles,
	cusolverDnHandle_t *cuSolverHandles, double *d_num_t, double *d_num_val, double** pointers)
{
	//size_t newcols = 0;
	//size_t cols = Xt.ncols(), rows = Xt.nrows();
	//int *remap = (int*)malloc(sizeof(int)*cols);
	////Create a mask
	//for (int i = 0; i < cols; ++i)
	//	if (curr_mask[i]) {
	//		remap[newcols] = i;
	//		newcols++;
	//	}

	size_t rows = rows = Xt.nrows();

	dim3 * dims = getMaxGrid(Xt.nrows(), newcols);

	//cud(cudaStreamSynchronize(streams[4]));
	//cud(cudaMemcpyAsync((int*)pointers[s.d_remap], remap, sizeof(int)*cols, cudaMemcpyHostToDevice, streams[4]));
	//cud(cudaStreamSynchronize(streams[4]));

	//Filter X matrices
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	deviceFilter <<< dims[0], dims[1], 0, streams[0] >>> (newcols, rows, Xt.values, (double*)pointers[s.d_Xt_filtered], d_remap);
	deviceFilter <<< dims[0], dims[1], 0, streams[1] >>> (newcols, rows, Xval.values, (double*)pointers[s.d_Xval_filtered], d_remap);

	//Filter X matrices
	cud(cudaStreamSynchronize(streams[2]));
	cud(cudaStreamSynchronize(streams[3]));
	cublas(cublasDgemm(cublasHandles[2], CUBLAS_OP_T, CUBLAS_OP_N, newcols, 1, rows, &one,
		(double*)pointers[s.d_Xt_filtered], rows, Yt.values, rows, &zero, (double*)pointers[s.d_XTdotY_t], newcols));
	cublas(cublasDgemm(cublasHandles[3], CUBLAS_OP_T, CUBLAS_OP_N, newcols, 1, rows, &one,
		(double*)pointers[s.d_Xval_filtered], rows, Yval.values, rows, &zero, (double*)pointers[s.d_XTdotY_val], newcols));

	//Synchronize for further gemm
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	cublas(cublasDgemm(cublasHandles[0], CUBLAS_OP_T, CUBLAS_OP_N, newcols, newcols, rows, &one,
		(double*)pointers[s.d_Xt_filtered], rows, (double*)pointers[s.d_Xt_filtered], rows, &zero, (double*)pointers[s.d_XTdotX_t], newcols));
	cublas(cublasDgemm(cublasHandles[1], CUBLAS_OP_T, CUBLAS_OP_N, newcols, newcols, rows, &one,
		(double*)pointers[s.d_Xval_filtered], rows, (double*)pointers[s.d_Xval_filtered], rows, &zero, (double*)pointers[s.d_XTdotX_val], newcols));

	if (newcols<10) {
		cud(cudaStreamSynchronize(streams[0]));
		cud(cudaStreamSynchronize(streams[1]));
		cublas(cublasDgetrfBatched(cublasHandles[0], newcols, (double**)pointers[s.d_matrices], newcols, (int*)pointers[s.d_piv], (int*)pointers[s.d_info], 2));
		cud(cudaStreamSynchronize(streams[0]));
		cublas(cublasDgetriBatched(cublasHandles[0], newcols, (const double **)pointers[s.d_matrices], newcols, 
			(int*)pointers[s.d_piv], (double**)pointers[s.d_inverted], newcols, (int*)pointers[s.d_info], 2));
	}
	else {
		cudaStream_t *streams_t = new cudaStream_t[2]{ streams[0], streams[4] };
		cudaStream_t *streams_val = new cudaStream_t[2]{ streams[1], streams[5] };
		cud(cudaStreamSynchronize(streams[0]));
		cud(cudaStreamSynchronize(streams[1]));
		cud(cudaStreamSynchronize(streams[4]));
		cud(cudaStreamSynchronize(streams[5]));
		//Synchronize again, this time for inversion
		thread th1 = thread(invPipelineCUDA, newcols, ref(streams_t), cuSolverHandles[0], (double*)pointers[s.d_XTdotX_t], ref((double*&)pointers[s.d_XTdotX_inv_t]));
		thread th2 = thread(invPipelineCUDA, newcols, ref(streams_val), cuSolverHandles[1], (double*)pointers[s.d_XTdotX_val], ref((double*&)pointers[s.d_XTdotX_inv_val]));
		th1.join(); th2.join();
		delete[] streams_t;
		delete[] streams_val;
	}

	//Gemm to get coefs
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	cublas(cublasDgemm(cublasHandles[0], CUBLAS_OP_N, CUBLAS_OP_N, newcols, 1, newcols, &one,
		(double*)pointers[s.d_XTdotX_inv_t], newcols, (double*)pointers[s.d_XTdotY_t], newcols, &zero,  (double*)pointers[s.d_coefs_t], newcols));
	cublas(cublasDgemm(cublasHandles[1], CUBLAS_OP_N, CUBLAS_OP_N, newcols, 1, newcols, &one,
		 (double*)pointers[s.d_XTdotX_inv_val], newcols,  (double*)pointers[s.d_XTdotY_val], newcols, &zero,  (double*)pointers[s.d_coefs_val], newcols));

	//Gemm to infere Ys
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	cublas(cublasDgemm(cublasHandles[0], CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, newcols, &one,
		(double*)pointers[s.d_Xt_filtered], rows,  (double*)pointers[s.d_coefs_t], rows, &zero,  (double*)pointers[s.d_Y_hat_t], rows));
	cublas(cublasDgemm(cublasHandles[1], CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, newcols, &one,
		 (double*)pointers[s.d_Xval_filtered], rows,  (double*)pointers[s.d_coefs_val], rows, &zero,  (double*)pointers[s.d_Y_hat_val], rows));

	//Get error
	dim3 * rowDim = getOneDims(rows);

	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	elementwise << < rowDim[0], rowDim[1], 0, streams[0] >> > ( (double*)pointers[s.d_Y_hat_t], Yt.values, rows,  (double*)pointers[s.diff_t], diff<double>());
	elementwise << < rowDim[0], rowDim[1], 0, streams[1] >> > ( (double*)pointers[s.d_Y_hat_val], Yval.values, rows,  (double*)pointers[s.diff_val], diff<double>());


	//Make it square
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	elementwise << < rowDim[0], rowDim[1], 0, streams[0] >> > ( (double*)pointers[s.diff_t],  (double*)pointers[s.diff_t], rows,  (double*)pointers[s.d_diff_t_sq], product<double>());
	elementwise << < rowDim[0], rowDim[1], 0, streams[1] >> > ( (double*)pointers[s.diff_val],  (double*)pointers[s.diff_val], rows,  (double*)pointers[s.d_diff_val_sq], product<double>());


	//Reduce 
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	reduce((double*)pointers[s.d_diff_t_sq], rows, sum<double>(), &d_num_t, streams[0]);
	reduce((double*)pointers[s.d_diff_val_sq], rows, sum<double>(), &d_num_val, streams[1]);
}

void cudaPrecomputeDenominators(Matrix& Yt, Matrix Yval, double &denom_t, double &denom_val, cudaStream_t *streams)
{
	size_t rows = Yt.rows;
	dim3 * rowDim = getOneDims(rows);
	double *d_denom_t, *d_denom_val;
	double *d_Yt_sq, *d_Yval_sq;
	cud(cudaMalloc(&d_Yt_sq, sizeof(double)*rows));
	cud(cudaMalloc(&d_Yval_sq, sizeof(double)*rows));

	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	elementwise << < rowDim[0], rowDim[1], 0, streams[0] >> > (Yt.values, Yt.values, rows, d_Yt_sq, product<double>());
	elementwise << < rowDim[0], rowDim[1], 0, streams[1] >> > (Yval.values, Yval.values, rows, d_Yval_sq, product<double>());

	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));

	cud(cudaMalloc((void **)&d_denom_t, sizeof(double)));
	cud(cudaMalloc((void **)&d_denom_val, sizeof(double)));

	reduce(d_Yt_sq, rows, sum<double>(), &d_denom_t, streams[0]);
	reduce(d_Yval_sq, rows, sum<double>(), &d_denom_val, streams[1]);
	cud(cudaStreamSynchronize(streams[0]));
	cud(cudaStreamSynchronize(streams[1]));
	cud(cudaMemcpy(&denom_t, d_denom_t, sizeof(double), cudaMemcpyDeviceToHost));
	cud(cudaMemcpy(&denom_val, d_denom_val, sizeof(double), cudaMemcpyDeviceToHost));
	cud(cudaFree(d_denom_t));
	cud(cudaFree(d_denom_val));
}

double** allocateDeviceMemory(size_t rows, size_t cols)
{
	double** pointers = (double**)malloc(sizeof(double*) * 23);

	cud(cudaMalloc((void**)&pointers[s.d_XTdotX_inv_t], sizeof(double)*cols*cols));
	cud(cudaMalloc((void**)&pointers[s.d_XTdotX_inv_val], sizeof(double)*cols*cols));
	cud(cudaMalloc((void**)&pointers[s.d_remap], sizeof(int)*cols));
	cud(cudaMalloc((void**)&pointers[s.d_Xt_filtered], sizeof(double)*cols*rows));
	cud(cudaMalloc((void**)&pointers[s.d_Xval_filtered], sizeof(double)*cols*rows));
	cud(cudaMalloc((void**)&pointers[s.d_XTdotX_t], sizeof(double)*cols*cols));
	cud(cudaMalloc((void**)&pointers[s.d_XTdotX_val], sizeof(double)*cols*cols));
	cud(cudaMalloc((void**)&pointers[s.d_XTdotY_t], sizeof(double)*cols));
	cud(cudaMalloc((void**)&pointers[s.d_XTdotY_val], sizeof(double)*cols));

	cud(cudaMalloc((void**)&pointers[s.d_matrices], 2 * sizeof(double *)));
	cud(cudaMalloc((void**)&pointers[s.d_inverted], 2 * sizeof(double *)));

	double ** matrices = new double*[2]{ pointers[s.d_XTdotX_t], pointers[s.d_XTdotX_val] };
	double ** inverted = new double*[2]{ pointers[s.d_XTdotX_inv_t], pointers[s.d_XTdotX_inv_val] };
	cud(cudaMemcpy(pointers[s.d_matrices], matrices, sizeof(double*) * 2, cudaMemcpyHostToDevice));
	cud(cudaMemcpy(pointers[s.d_inverted], inverted, sizeof(double*) * 2, cudaMemcpyHostToDevice));
	
	cud(cudaMalloc((void**)&pointers[s.d_piv], 2 * cols * sizeof(int)));
	cud(cudaMalloc((void**)&pointers[s.d_info], 2 * sizeof(int)));

	cud(cudaMalloc((void**)&pointers[s.d_coefs_t], sizeof(double)*cols));
	cud(cudaMalloc((void**)&pointers[s.d_coefs_val], sizeof(double)*cols));
	cud(cudaMalloc((void**)&pointers[s.d_Y_hat_t], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.d_Y_hat_val], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.diff_t], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.diff_val], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.d_diff_t_sq], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.d_diff_val_sq], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.d_Yt_sq], sizeof(double)*rows));
	cud(cudaMalloc((void**)&pointers[s.d_Yval_sq], sizeof(double)*rows));

	delete[] matrices;
	delete[] inverted;

	return pointers;
}

void deallocateDeviceMemory(void** allocations)
{
	for (int i = 0; i < 23; ++i) cud(cudaFree(allocations[i]));
}

void createCUDAWorkspace(cudaStream_t * &streams, cusolverDnHandle_t * &cusdolverHandles, cublasHandle_t * &cublasHandles)
{
	streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 6);
	cusdolverHandles = (cusolverDnHandle_t*)malloc(sizeof(cusolverDnHandle_t) * 2);
	cublasHandles = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 4);
	for (int i = 0; i < 6; ++i) {
		streams[i] = cudaStream_t();
		cudaStreamCreate(&streams[i]);
		if (i<4) {
			cublasHandles[i] = cublasHandle_t();
			cublasCreate_v2(&cublasHandles[i]);
			cublasSetStream_v2(cublasHandles[i], streams[i]);
			if (i<2) {
				cusdolverHandles[i] = cusolverDnHandle_t();
				cusolverDnCreate(&cusdolverHandles[i]);
				cusolverDnSetStream(cusdolverHandles[i], streams[i]);
			}
		}
	}
}

void destroyCUDAWorkspace(cudaStream_t * &streams, cusolverDnHandle_t * &cusdolverHandles, cublasHandle_t * &cublasHandles) 
{
	for (int i = 0; i < 6; ++i) {
		if (i < 4) {
			if (i < 2)
				cusolverDnDestroy(cusdolverHandles[i]);
			cublasDestroy_v2(cublasHandles[i]);
		}
		cudaStreamDestroy(streams[i]);
	}
	free(cublasHandles);
	free(cusdolverHandles);
	free(streams); 
}

