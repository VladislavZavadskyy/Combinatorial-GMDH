#ifndef TESTS_CUH_
#define TESTS_CUH_

#include "dependencies.cuh"
#include "matrix.cuh"

void testDot(int starting_rows, int starting_cols, int starting_common_dim,
	int max_rows, int max_cols, int max_common_dim,
	int row_step, int col_step, int common_dim_step,
	double epsilon = 1e-3);
void testDot(int starting_dims, int max_dims, int dim_step);
void testDot();

void testEqual(int starting_rows, int starting_cols,
	int max_rows, int max_cols, int row_step, int col_step,
	double eps = 1e-4, int runs_per_dim = 3);
void testEqual(int starting_dims, int max_dims, int dim_step);
void testEqual();

void testTranspose(int starting_rows, int starting_cols,
	int max_rows, int max_cols,
	int row_step, int col_step, double eps=1e-4);
void testTranspose(int starting_dims, int max_dims, int dim_step);
void testTranspose();

void testInverse(int starting_dims, int max_dims, int dim_step);
void testInverse();
#endif