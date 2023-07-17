#include <iostream>
#include <vector>

#include "..\text.cpp"

using namespace std;


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
    for(auto& i : tfidf.get_vocabulary()){
        if(i.first != test_result.at(i.second))
            return false;
    }
    return true;
}

bool CountVectorizer_vocabulary_test(){
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
    CountVectorizer cntvec = CountVectorizer("en");
    cntvec.fit(corpus);
    for(auto& i : cntvec.get_vocabulary()){
        if(i.first != test_result.at(i.second))
            return false;
    }
    return true;
}

SparseMatrix<double> generate_example_spm(vector<vector<double>> data){
    vector<Triplet<double>> tripletList;
    for(size_t i=0; i<data.size(); ++i)
        for(size_t j=0; j<data[i].size(); ++j)
            tripletList.push_back(Triplet<double>(i, j, data[i][j]));
    SparseMatrix<double> mat(data.size(), data[0].size());
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    mat.makeCompressed();
    mat.finalize();
    return mat;
}

bool tfidf_transform_test(){
    vector<vector<double>> data = {{1.0, 0}, {0.8154290342094731, 0.638763577291385}, {0, 0}, {1.0, 0}};
    SparseMatrix<double> spm_res = generate_example_spm(data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    TFIDF tfidf = TFIDF("en");
    auto x = *tfidf.fit(corpus).transform(corpus);
    if(x.size() != spm_res.size() && x.rows() == spm_res.rows() && x.cols() == spm_res.cols())
        return false;
    for(size_t i=0; i<x.rows(); ++i)
        for(size_t j=0; j<x.cols(); ++j)
            if(x.coeffRef(i, j) != spm_res.coeffRef(i, j))
                return false;
    return true;
}

bool CountVectorizer_transform_test(){
    vector<vector<double>> data = {{1.0, 0}, {2.0, 1.0}, {0, 0}, {1.0, 0}};
    SparseMatrix<double> spm_res = generate_example_spm(data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    CountVectorizer cntvec = CountVectorizer("en");
    auto x = *cntvec.fit(corpus).transform(corpus);
    if(x.size() != spm_res.size() && x.rows() == spm_res.rows() && x.cols() == spm_res.cols())
        return false;
    for(size_t i=0; i<x.rows(); ++i)
        for(size_t j=0; j<x.cols(); ++j)
            if(x.coeffRef(i, j) != spm_res.coeffRef(i, j))
                return false;
    return true;
}

bool tfidf_fit_transform_test(){
    vector<vector<double>> data = {{1.0, 0}, {0.8154290342094731, 0.638763577291385}, {0, 0}, {1.0, 0}};
    SparseMatrix<double> spm_res = generate_example_spm(data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    TFIDF tfidf = TFIDF("en");
    auto x = tfidf.fit_transform(corpus);
    SparseMatrix<double>& res_x = *x;
    if(res_x.size() != spm_res.size() && res_x.rows() == spm_res.rows() && res_x.cols() == spm_res.cols())
        return false;
    for(size_t i=0; i<res_x.rows(); ++i)
        for(size_t j=0; j<res_x.cols(); ++j)
            if(res_x.coeffRef(i, j) != spm_res.coeffRef(i, j))
                return false;
    return true;
}

bool CountVectorizer_fit_transform_test(){
    vector<vector<double>> data = {{1.0, 0}, {2.0, 1.0}, {0, 0}, {1.0, 0}};
    SparseMatrix<double> spm_res = generate_example_spm(data);
    vector<string> corpus {
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    };
    CountVectorizer cntvec = CountVectorizer("en");
    auto x = cntvec.fit_transform(corpus);
    SparseMatrix<double>& res_x = *x;
    if(res_x.size() != spm_res.size() && res_x.rows() == spm_res.rows() && res_x.cols() == spm_res.cols())
        return false;
    for(size_t i=0; i<res_x.rows(); ++i)
        for(size_t j=0; j<res_x.cols(); ++j)
            if(res_x.coeffRef(i, j) != spm_res.coeffRef(i, j))
                return false;
    return true;
}

int main(){
    assert(get_feature_names_out_test() && "Failed test: tokenizer_test");
    assert(tfidf_vocabulary_test() && "Failed test: tfidf_test");
    assert(tfidf_transform_test() && "Failed test: tfidf_transform_test");
    assert(tfidf_fit_transform_test() && "Failed test: tfidf_fit_transform_test");
    assert(CountVectorizer_vocabulary_test() && "Failed test: CountVectorizer_vocabulary_test");
    assert(CountVectorizer_transform_test() && "Failed test: CountVectorizer_transform_test");
    assert(CountVectorizer_fit_transform_test() && "Failed test: CountVectorizer_fit_transform_test");
    cout << "All tests passed" << endl;
    return 0;
}