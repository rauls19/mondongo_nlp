#include <iostream>
#include <vector>

#include "..\text.cpp"

using namespace std;

bool sparse_creation_test_1(){
    // Correct result
    vector<double> test_data = {1, 1, 1, 1, 1, 1};
    vector<size_t> test_idx_tuple = {0, 0, 0, 1, 1, 1};
    vector<size_t> test_col_tuple = {0, 1, 0, 2, 3, 1};

    // Init test data
    vector<size_t> indptr = {0, 3, 6};
    vector<size_t>column = {0, 1, 0, 2, 3, 1};
    vector<double> data =  {1, 1, 1, 1, 1, 1};
    csr_matrix csr_matrix_test = csr_matrix(indptr, column, data);
    size_t j = 0;
    for(auto& i : csr_matrix_test.spm){
        if ((get<0>(i.index) != test_idx_tuple.at(j)) || (get<1>(i.index) != test_col_tuple.at(j)) || (i.value != test_data.at(j))){
            return false;
        }
        j++;
    }
    return true;
}
bool sparse_creation_test_2(){
    // Correct result
    vector<double> test_data = {1, 2, 3, 4, 5};
    vector<size_t> test_idx_tuple = {0, 0, 1, 2, 2};
    vector<size_t> test_col_tuple = {0, 1, 2, 0, 2};

    // Init test data
    vector<size_t> indptr = {0, 2, 3, 5};
    vector<size_t>column = {0, 1, 2, 0, 2};
    vector<double> data =  {1, 2, 3, 4, 5};
    csr_matrix csr_matrix_test = csr_matrix(indptr, column, data);
    size_t j = 0;
    for(auto& i : csr_matrix_test.spm){
        if ((get<0>(i.index) != test_idx_tuple.at(j)) || (get<1>(i.index) != test_col_tuple.at(j)) || (i.value != test_data.at(j))){
            return false;
        }
        j++;
    }
    return true;
}

bool get_feature_names_out_test(){
    // Correct result
    vector<vector<string>> test_result = {
        {"document"},
        {"document", "second", "document"},
        {""},
        {"document"}
    };

    // Init test data
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    Tokenizer tokenizer = Tokenizer();
    auto tokens = tokenizer.fit_transform(corpus);
    for(size_t i=0;i<tokens.size(); i++)
        for(size_t j=0; j<tokens.at(i).size(); j++)
            if(tokens.at(i).at(j) != test_result.at(i).at(j))
                return false;
    return true;
}

bool tfidf_vocabulary_test(){
    // Correct result
    vector<string> test_result = {
        "document", "second"
    };

    // Init test data
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    TFIDF tfidf = TFIDF("en");
    tfidf.fit(corpus);
    size_t j = 0;
    for(auto& i : tfidf.vocabulary){
        if(i.first != test_result.at(j))
            return false;
        j++;
    }
    return true;
}

bool tfidf_transform_test(){
   vector<size_t> indptr = {0, 1, 3, 3, 4};
   vector<double> data = {1.0, 0.638763577291385, 0.8154290342094731, 1.0};
   vector<size_t> indices = {0, 1, 0, 0};
   csr_matrix spm_res = csr_matrix(indptr, indices, data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    TFIDF tfidf = TFIDF("en");
    auto x = tfidf.fit(corpus).transform(corpus);
    if(x.spm.size() != spm_res.spm.size())
        return false;
    for(size_t i=0; i<spm_res.spm.size(); i++)
        if((x.spm.at(i).index != spm_res.spm.at(i).index) || 
            (x.spm.at(i).value != spm_res.spm.at(i).value))
                return false;
    return true;   
}

bool tfidf_fit_transform_test(){
   vector<size_t> indptr = {0, 1, 3, 3, 4};
   vector<double> data = {1.0, 0.638763577291385, 0.8154290342094731, 1.0};
   vector<size_t> indices = {0, 1, 0, 0};
   csr_matrix spm_res = csr_matrix(indptr, indices, data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    TFIDF tfidf = TFIDF("en");
    auto x = tfidf.fit_transform(corpus);
    if(x.spm.size() != spm_res.spm.size())
        return false;
    for(size_t i=0; i<spm_res.spm.size(); i++)
        if((x.spm.at(i).index != spm_res.spm.at(i).index) || 
            (x.spm.at(i).value != spm_res.spm.at(i).value))
                return false;
    return true;   
}

int main(){
    assert(sparse_creation_test_1() && "Failed test: sparse_creation_test_1");
    assert(sparse_creation_test_2() && "Failed test: sparse_creation_test_2");
    assert(get_feature_names_out_test() && "Failed test: tokenizer_test");
    assert(tfidf_vocabulary_test() && "Failed test: tfidf_test");
    assert(tfidf_transform_test() && "Failed test: tfidf_transform_test");
    assert(tfidf_fit_transform_test() && "Failed test: tfidf_fit_transform_test");
    cout << "All tests passed" << endl;
    return 0;
}