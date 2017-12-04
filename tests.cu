#include "tests.cuh"

void testDot(int starting_rows, int starting_cols, int starting_common_dim,
			 int max_rows, int max_cols, int max_common_dim,
			 int row_step, int col_step, int common_dim_step,
			 double epsilon)
{
	#ifdef WINDOWS
	HANDLE  hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	#endif
	for (int ncols = starting_cols; ncols <= max_cols; ncols+=col_step)
		for (int nrows = starting_rows; nrows <= max_rows; nrows+=row_step)
			for (int interdim = starting_common_dim; interdim <= max_common_dim; interdim+=common_dim_step)
			{
				Matrix A(nrows, interdim, indexInitializer<double>);
				Matrix B(interdim, ncols, indexInitializer<double>);
				Matrix ref = A.dot(B);
				ref.cuda();
				A.cuda();  
				B.cuda();
				Matrix C = A.dot(B);
				#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 8);
				#endif
				printf("(%d,%d)x(%d,%d) ", nrows, interdim, interdim, ncols);
				if (ref.equal(C, epsilon)) {
					#ifdef WINDOWS
					SetConsoleTextAttribute(hConsole, 2);
					#endif
					printf("PASS\n");
				} else {
					#ifdef WINDOWS
					SetConsoleTextAttribute(hConsole, 4);
					#endif
					printf("FAIL\n");
					#ifdef WINDOWS
					SetConsoleTextAttribute(hConsole, 8);
					#endif
					printf("Reference:\n");
					ref.print();
					printf("Device result:\n");
					C.print();
					return;
				}
			}
}

void testDot(int starting_dims, int max_dims, int dim_step)
{
	testDot(starting_dims, starting_dims, starting_dims,
		max_dims, max_dims, max_dims, dim_step, dim_step, dim_step);
}

void testDot()
{
	testDot(2, 1024, 3);
}

void testEqual(int starting_rows, int starting_cols,
			   int max_rows, int max_cols,
			   int row_step, int col_step, 
			   double eps, int runs_per_dim)
{
#ifdef WINDOWS
	HANDLE  hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
	for (int ncols = starting_cols; ncols <= max_cols; ncols += col_step)
	for (int nrows = starting_rows; nrows <= max_rows; nrows += row_step) 
	for (int run = 0; run <= runs_per_dim; run++) {
		Matrix A(ncols, nrows, indexInitializer<double>);
		Matrix B(A);
		srand(time(NULL));

		int diff = (run == 0) ? 0 : rand() % A.ncols()*A.nrows();
		for (int i = 0; i < diff; ++i)
			B.seti(rand() % A.nrows(), rand() % A.ncols(), .1);

		bool ref = A.equal(B, eps);
		A.cuda(); B.cuda();
#ifdef WINDOWS
		SetConsoleTextAttribute(hConsole, 8);
		printf("(%d,%d) ", nrows, ncols);
#endif
		bool equality = A.equal(B, eps);
		if (ref==equality) {
#ifdef WINDOWS
			SetConsoleTextAttribute(hConsole, 2);
#endif
			printf("PASS\n");
		}
		else {
			#ifdef WINDOWS
			SetConsoleTextAttribute(hConsole, 4);
			#endif
			printf("FAIL\n");
			#ifdef WINDOWS
			SetConsoleTextAttribute(hConsole, 8);
			#endif
			printf("A\n");
			A.print();
			printf("B\n");
			B.print();
			printf("Device says: %d, host contradicts with %d", equality, ref);
			return;
		}
	}
}

void testEqual(int starting_dims, int max_dims, int dim_step)
{
	testEqual(starting_dims, starting_dims, max_dims, max_dims, dim_step, dim_step);
}

void testEqual()
{
	testEqual(1, 512, 1);
}

void testTranspose(int starting_rows, int starting_cols,
				   int max_rows, int max_cols,
				   int row_step, int col_step,
				   double eps)
{
#ifdef WINDOWS
	HANDLE  hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
	for (int ncols = starting_cols; ncols <= max_cols; ncols += col_step)
		for (int nrows = starting_rows; nrows <= max_rows; nrows += row_step){
			Matrix A(ncols, nrows, indexInitializer<double>);
			Matrix ref = A.T();
			ref.cuda(); A.cuda();
			Matrix A_t = A.T();

#ifdef WINDOWS
			SetConsoleTextAttribute(hConsole, 8);
#endif
			printf("(%d,%d) ", nrows, ncols);
			if (A_t.equal(ref)) {
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 2);
#endif
				printf("PASS\n");
			}
			else {
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 4);
#endif
				printf("FAIL\n");
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 8);
#endif
				printf("A\n");
				A.print();
				printf("Reference\n");
				ref.print();
				return;
			}
		}
}

void testTranspose(int starting_dims, int max_dims, int dim_step)
{
	testTranspose(starting_dims, starting_dims, max_dims, max_dims, dim_step, dim_step);
}

void testTranspose()
{
	testTranspose(1, 512, 1);
}


void testInverse(int starting_dims, int max_dims, int dim_step)
{
#ifdef WINDOWS
	HANDLE  hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
	for (int dim = starting_dims; dim <= max_dims; dim += dim_step) {
			Matrix A(dim, dim, normalInitializer<double>);
			Matrix Copy(A);
			Matrix Orig(A);
			
			Matrix ref = Copy.inv();
			ref.cuda(); 
			A.cuda();
			Matrix inv = A.inv();

#ifdef WINDOWS
			SetConsoleTextAttribute(hConsole, 8);
#endif
			printf("(%d,%d) ", dim, dim);
			if (A.equal(ref)) {
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 2);
#endif
				printf("PASS\n");
			}
			else {
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 4);
#endif
				printf("FAIL\n");
#ifdef WINDOWS
				SetConsoleTextAttribute(hConsole, 8);
#endif
				printf("\nOriginal:");
				Orig.print();
				printf("\nA");
				A.print();
				printf("\nReference");
				ref.print();
				return;
			}
		}
}

void testInverse()
{
	testInverse(1, 512, 1);
}