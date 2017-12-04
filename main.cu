#include "utils.cuh"
#include "ops.cuh"
#include "matrix.cuh"
#include "tests.cuh"

const size_t BATCH_SIZE = 10000;
using namespace std;

void GMDHtask(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix& Y, 
	size_t start, size_t end, bool* &best_mask, double &bestCrit, bool verbose, int device,
	double denom_t, double denom_val)
{
	bool * curr_mask = (bool*)malloc(sizeof(bool)*X.ncols());
	double inner_crit, outer_crit;
	bestCrit = INFINITY;

	if (Xt._isCuda()) {
		cudaStream_t * streams; //Should be at least 6
		cusolverDnHandle_t * cusdolverHandles; //Should be at least 2
		cublasHandle_t * cublasHandles; //Should be at least 4
		double** pointers;
		double *d_numerators_t, *d_numerators_val;
		double *numerators_t, *numerators_val;
		int *remaps, *d_remaps, *newcols_s;

		createCUDAWorkspace(streams, cusdolverHandles, cublasHandles);
		pointers = allocateDeviceMemory(Xt.nrows(), Xt.ncols());
		cud(cudaSetDevice(device));

		cud(cudaMalloc(&d_numerators_t, sizeof(double*)*(end - start)));
		cud(cudaMalloc(&d_numerators_val, sizeof(double*)*(end - start)));

		remaps = (int*)malloc(sizeof(int)*BATCH_SIZE*Xt.ncols());
		newcols_s = (int*)malloc(sizeof(int)*BATCH_SIZE);
		cud(cudaMalloc((void**)&d_remaps, sizeof(int)*BATCH_SIZE*Xt.ncols()));

		for (size_t i = start; i < end; ++i) {
			if ((i-start)%BATCH_SIZE==0){
				int num_batch = min(BATCH_SIZE, end - i);
				for (int k = 0; k < num_batch; ++k)
				{
					int newcols = 0;
					for (int j = 0; j < X.ncols(); ++j)
						curr_mask[j] = (i + k >> j) & 1;
					for (int j = 0; j < X.ncols(); ++j)
						if (curr_mask[j]) {
							remaps[X.ncols()*k + newcols] = j;
							newcols++;
						} 
					newcols_s[k] = newcols;
				}
				cud(cudaMemcpyAsync(d_remaps, remaps, sizeof(int)*num_batch*Xt.ncols(), 
					cudaMemcpyHostToDevice, streams[0]));
			}
			int remap_ind = i - start - floor(double(i - start)/BATCH_SIZE)*BATCH_SIZE;
			CUDAPipeline(Xt, Xval, Yt, Yval, X, Y, &d_remaps[remap_ind*X.ncols()], newcols_s[remap_ind],
				streams, cublasHandles, cusdolverHandles, &d_numerators_t[i - start], 
				&d_numerators_val[i - start], pointers);
		}

		numerators_t = (double*)malloc(sizeof(double)*(end - start));
		numerators_val = (double*)malloc(sizeof(double)*(end - start));

		cud(cudaMemcpyAsync(numerators_t, d_numerators_t, sizeof(double)*(end - start), cudaMemcpyDeviceToHost, streams[0]));
		cud(cudaMemcpyAsync(numerators_val, d_numerators_val, sizeof(double)*(end - start), cudaMemcpyDeviceToHost, streams[1]));
		cud(cudaStreamSynchronize(streams[0]));
		cud(cudaStreamSynchronize(streams[1]));

		for (int i = start; i < end; ++i) {
			outer_crit = numerators_val[i-start] / denom_val;
			if (outer_crit < bestCrit) {
				bestCrit = outer_crit;
				for (int j = 0; j < X.ncols(); ++j)
					curr_mask[j] = (i >> j) & 1;
				memcpy(best_mask, curr_mask, sizeof(bool)*X.ncols());
			}
		}

		destroyCUDAWorkspace(streams, cusdolverHandles, cublasHandles);
		deallocateDeviceMemory((void**)pointers);
		cud(cudaFree(d_remaps));
		free(newcols_s);
		free(remaps);
	} else {
		for (int i = start; i < end; ++i)
		{
			for (int j = 0; j < X.ncols(); ++j)
				curr_mask[j] = (i >> j) & 1;

			Matrix filtered = Xt.filterCols(curr_mask);
			Matrix filterdVal = Xval.filterCols(curr_mask);
			Matrix inferred = move(leastSquares(filtered, Yt));

			outer_crit = computeCriterion(filterdVal, Yval, inferred);

			if (verbose) {
				printf("Mask:\n");
				for (int j = 0; j < Xt.ncols(); ++j)
					printf("%d ", curr_mask[j]);
				printf("\n");

				printf("Infered coefficients:");
				inferred.print();

				printf("Outer criterion value: %.2e\n", outer_crit);
				inner_crit = computeCriterion(filtered, Yt, inferred);
				printf("Inner criterion value: %.2e\n", inner_crit);

				double unbiasedness = computeUnbiasedness(filtered, filterdVal, Yt, Yval, X.filterCols(curr_mask), Y);
				printf("Unbiassedness criterion value: %.5e\n", unbiasedness);
			}

			if (outer_crit < bestCrit) {
				bestCrit = outer_crit;
				memcpy(best_mask, curr_mask, sizeof(bool)*X.ncols());
			}
		}
	}
	free(curr_mask);
}

unsigned long long counter = 0;
void argParse(int argc, char** argv, bool &cuda, int &numThreads, 
			  char* &Xname, char* &Yname, bool &verbose)
{
	cuda = false;
	verbose = false;
	Xname = "C:\\Data\\X15.csv";
	Yname = "C:\\Data\\Y15.csv";
	numThreads = 8;

	for (int i = 0; i < argc; ++i)
	{
		if (!strcmp(argv[i], "cpu")) cuda = false;
		if (!strcmp(argv[i], "verbose")) {
			if (cuda) printf("Due to efficiency concerns, CUDA mode can only be silent or silent.\n");
			else verbose = true;
		}
		if (!strcmp(argv[i], "x")) Xname = argv[i + 1];
		if (!strcmp(argv[i], "y")) Yname = argv[i + 1];
		if (!strcmp(argv[i], "nt")) numThreads = stoi(argv[i+1]);
	}
}

#define TEST0
int main(int argc, char** argv)
{
#ifndef TEST
	bool cuda, verbose;
	int num_threads;
	char *Xname, *Yname;
	argParse(argc, argv, cuda, num_threads, Xname, Yname, verbose);
	printf("Current settings:\n");
	printf("\tCUDA: %s\n",(cuda) ? "enabled" : "disabled");
	printf("\tFile paths: \n\t  X: %s \n\t  Y: %s\n", Xname, Yname);
	printf("\tNumber of threads: %d\n", num_threads);
	printf("\t%s mode\n\n", (verbose) ? "Verbose" : "Silent");

	vector<thread> threads(num_threads);
	Matrix Xmeta = csv2mat(Xname);
	Xmeta = Xmeta.insertCol(-1, onesInitializer<double>);
	Matrix Ymeta = csv2mat(Yname);
	int dev_count;
	if (cuda){
		Xmeta.cuda();
		Ymeta.cuda();
		cudaGetDeviceCount(&dev_count);
	}
	size_t meta_split = Xmeta.nrows()*0.75;
	Matrix X = Xmeta.sub(0, 0, Xmeta.ncols(), meta_split);
	Matrix Xtest = Xmeta.sub(meta_split, 0, Xmeta.ncols(), Xmeta.nrows() - meta_split);
	Matrix Y = Ymeta.sub(0, 0, Ymeta.ncols(), meta_split);
	Matrix Ytest = Ymeta.sub(meta_split, 0, Ymeta.ncols(), Ymeta.nrows() - meta_split);

	//X.print(); Xtest.print(); Y.print(); Ytest.print();
	if(X.nrows()!=Y.nrows()){
		printf("Number of datapoints in X and Y doesn't match. Aborting.");
		return;
	}
	const int split_point = X.nrows() / 2;
	Matrix Xt = X.sub(0, 0, X.ncols(), split_point);
	Matrix Yt = Y.sub(0, 0, Y.ncols(), split_point);
	Matrix Xval = X.sub(split_point, 0, X.ncols(), split_point);
	Matrix Yval = Y.sub(split_point, 0, Y.ncols(), split_point);
	//Xt.print(); Yt.print(); Xval.print(); Yval.print();
	const int num_masks = pow(2, X.ncols());

	//bool ** masks = generateMasks(X.ncols());
	
	bool ** best_masks = (bool**)malloc(sizeof(bool**)*num_threads);
	for (int i = 0; i < num_threads; ++i)
		best_masks[i] = (bool*)malloc(sizeof(bool*)*X.ncols());
	double * best_crits = (double*)malloc(sizeof(double*)*num_threads);
	size_t masks_per_thread = num_masks / num_threads;
	int threads_per_device;

	double denom_t, denom_val;
	if (cuda) {
		threads_per_device = CEIL(num_threads, dev_count);
		cudaStream_t * streams = new cudaStream_t[2];
		cud(cudaStreamCreate(&streams[0]));
		cud(cudaStreamCreate(&streams[1]));
		cudaPrecomputeDenominators(Yt, Yval, denom_t, denom_val, streams);
		cud(cudaStreamDestroy(streams[0]));
		cud(cudaStreamDestroy(streams[1]));
		delete[] streams;
	}
	using namespace chrono;
	long t0 = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()).count();
	int device;
	for (int i = 0; i < num_threads; ++i)
	{
		if (cuda) device = i / threads_per_device;
		size_t start = (i == 0) ? 1 : i*masks_per_thread+1;
		size_t end = (i == (num_threads - 1)) ? num_masks : (i + 1)*masks_per_thread+1;
		threads[i] = thread(GMDHtask, ref(Xt), ref(Xval), ref(Yt), ref(Yval), ref(Xmeta), ref(Ymeta),
			start, end, ref(best_masks[i]), ref(best_crits[i]), verbose, device, denom_t, denom_val);
	}
	for (int j = 0; j < num_threads; ++j) threads[j].join();
	long t1 = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()).count();

	bool* best_mask = nullptr;
	double best_crit = INFINITY;
	for (int i = 0; i < num_threads; ++i)
		if (best_crit > best_crits[i])
		{
			best_crit = best_crits[i];
			best_mask = best_masks[i];
		}

	Matrix Xfiltered = X.filterCols(best_mask);
	Matrix bestShot = leastSquares(Xfiltered, Y);
	const double crit_value = computeCriterion(Xfiltered, Y, bestShot);

	printf("-------Best results-------\n");
	printf("Masque: ");
	printMask(best_mask, X.ncols());
	printf("Coefficients:");
	bestShot.print();
	printf("Criterion value: %.5e\n", crit_value);

	// ReSharper disable CppMsExtBindingRValueToLvalueReference
	const double unbiasedness = computeUnbiasedness(Xt.filterCols(best_mask), 
		Xval.filterCols(best_mask), Yt, Yval, Xmeta.filterCols(best_mask), Ymeta);
	printf("Unbiassedness criterion value: %.5e\n", unbiasedness);

	printf("\nBaseline:");
	Matrix baselineCoefs = leastSquares(X, Y);
	const double baseline_crit = computeCriterion(X, Y, baselineCoefs);
	baselineCoefs.print();
	printf("Criterion value: %.5e\n", baseline_crit);
	const double b_line_unbiasedness = computeUnbiasedness(Xt, Xval, Yt, Yval, Xmeta, Ymeta);
	printf("Unbiassedness criterion value: %.5e\n", b_line_unbiasedness);

	printf("Time elapsed %.2f seconds.\n", (double)(t1 - t0)/1000);

	free(best_masks);
	free(best_crits);

	cin.ignore();
	return 0;
#else
	testInverse();
	cin.ignore();
#endif
}
