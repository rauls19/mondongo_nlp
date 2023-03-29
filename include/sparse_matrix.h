#ifndef csr_matrix_h
#define csr_matrix_h

#include <vector>
#include <tuple>

using namespace std;

class csr_matrix{
    private:
    // TODO: Define here funtions to get indptr ?? etc. ?
    void create_sparse_matrix(vector<double> data, int row, int col);
    void create_sparse_matrix(vector<size_t> indptr, vector<size_t> indices, vector<double> data);
    void create_sparse_matrix(vector<vector<double>> matrix);

    public:
    struct cell {
        tuple<size_t, size_t> index;
        double value;
    };

    vector<cell> spm;

    csr_matrix();
    csr_matrix(int rows, int cols);
    csr_matrix(vector<double> data, int row, int col);
    csr_matrix(vector<size_t> indptr, vector<size_t> indices, vector<double> data);
    csr_matrix(vector<vector<double>> matrix);

    // csr_matrix tocsr();
    // csr_matrix toarray();

};

#endif