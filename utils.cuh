#ifndef UTILS_CUH_
#define UTILS_CUH_

#define cud(res) { if (res != cudaSuccess) { printf("CUDA Error : %s %d : ERR %s\n", __FILE__, __LINE__, cudaGetErrorName(res)); } }
#define cublas(res) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s %d : ERR %d\n", __FILE__, __LINE__, int(res)); } }
#define cusolve(res) { if (res != CUSOLVER_STATUS_SUCCESS) { printf("cuSolver Error : %s %d : ERR %d\n", __FILE__, __LINE__, int(res)); } }

#include "dependencies.cuh"
#include "matrix.cuh"
class Matrix;
Matrix computeY(Matrix& X, Matrix& coefs, bool addNoise);
double computeCriterion(Matrix& X, Matrix& Y, Matrix& inferedCoefs);
Matrix leastSquares(Matrix& X, Matrix& Y);
bool** generateMasks(size_t cols);
void printMasks(bool** masks, size_t cols, bool print_sample);
void printMask(bool* mask, size_t cols);
void deleteMasks(bool** masks, size_t cols);
std::vector<std::string> split(const char *str, char c = ',');
std::vector<double> parseDoubleCSVString(const char* s);
std::vector<std::vector<double>> parseCSV(char * path);
Matrix vector2Matrix(std::vector<std::vector<double>> vector);
Matrix csv2mat(char * path);
double computeUnbiasedness(Matrix& Xt, Matrix& Xval, Matrix& Yt, Matrix& Yval, Matrix& X, Matrix &Y);

#endif