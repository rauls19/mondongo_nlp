#include <iostream>
#include <vector>
#include <cassert>

#include "..\include\sparse_matrix.h"

using namespace std;

void csr_matrix::create_sparse_matrix(vector<double> data, int row, int col){}

void csr_matrix::create_sparse_matrix(vector<size_t> indptr, vector<size_t> indices, vector<double> data){
    assert(indices.size() == data.size() && "Columns and Values must have the same size");
    size_t idx = 0;
    for(size_t i=0; i<indptr.size()-1; i++){
        size_t n_val_row = indptr.at(i+1) - indptr.at(i);
        for(size_t j=indptr.at(i); j<n_val_row + indptr.at(i); j++)
            this->spm.push_back(csr_matrix::cell{make_tuple(idx, indices.at(j)), data.at(j)});
        idx++;
    }
}

void csr_matrix::create_sparse_matrix(vector<vector<double>> matrix){}

csr_matrix::csr_matrix(){}
csr_matrix::csr_matrix(int row, int col){}
csr_matrix::csr_matrix(vector<double> data, int row, int col){}
csr_matrix::csr_matrix(vector<size_t> indptr, vector<size_t> indices, vector<double> data){
    create_sparse_matrix(indptr, indices, data);
}
csr_matrix::csr_matrix(vector<vector<double>> matrix){}
