#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <cassert>
#include <set>
#include <map>
#include <execution>
#include <chrono>
#include <thread>
#include <mutex>
#include <numeric>
#include <future>

#include "..\..\third_party\eigen3\Eigen\Sparse"
#include "..\..\third_party\eigen3\Eigen\SparseCore"
#include "..\..\third_party\eigen3\Eigen\Dense"

using namespace Eigen;
using namespace std;

// TODOs:
// TODO: Tokenizer takes forever
// TODO: Clearn includes
// TODO: pointers etc
// TODO: put everything in a well format, using OOP?, virtual?, ...?
// TODO: make something to be able to execute it, like cmake or whatever


class Tokenizer{
    private:
    vector<string> en_sw;
    const vector<string> valid_lang = {"en"};

    void load_stopwords(string lang){
        ifstream en_stopwords; en_stopwords.open("../stopwords_"+lang+".txt");
        if(en_stopwords){
            string line;
            while(getline(en_stopwords, line)){
                en_sw.push_back(line);
            }
        }
        else{
            cerr << "Could not load English stopwords" << endl;
        }
    }

    public:
    Tokenizer(string lang="en"){
        assert(find(valid_lang.begin(), valid_lang.end(), lang) != valid_lang.end() && "Stopwords requested are not supported");
        load_stopwords(lang);
    }
    
    vector<vector<string>> fit_transform(const vector<string>& sentences){
        vector<vector<string>> tokens;
        tokens.reserve(sentences.size());
        for(const auto& line : sentences){
            string lowercase_line(line);
            lowercase_line.erase(remove_if(lowercase_line.begin(), lowercase_line.end(), ::ispunct), lowercase_line.end());
            transform(lowercase_line.begin(), lowercase_line.end(), lowercase_line.begin(), ::tolower);
            istringstream is(lowercase_line);
            string aux;
            vector<string> tok;
            while(getline(is, aux, ' ')){
                if(!(find(en_sw.begin(), en_sw.end(), aux) != en_sw.end())){
                    tok.push_back(move(aux));
                }
            }
            tokens.push_back(move(tok));
        }
        return tokens;
    }

    void visualize(vector<vector<string>> &tokens) const{
        for(const auto& i : tokens){
            for(const auto& j : i){
                cout << j + " ";
            }
            cout << "\n";
        }
    }
};


class TFIDF{
    private:
    string sw;
    vector<unordered_map<string, double>> tf;
    SparseMatrix<double> _tf_idf_spm;


    void create_map_vocabulary(const set<string>& vocab_keys){
        size_t i = 0;
        // Parallel?
        for(const auto& word : vocab_keys){
            vocabulary[word] = i;
            reverse_vocabulary[i] = word;
            i++;
        }
    }

    SparseMatrix<double> convert_to_sparsematrix(size_t col_dim){
        SparseMatrix<double> spm(tf.size(), col_dim);
        vector<Triplet<double>> triplets; // Store nonzero entries
        #pragma omp parallel for schedule(guided) reduction(+:triplets)
        {
            for(size_t i=0; i<tf.size(); i++){
                const auto& sentence = tf[i];
                for(const auto&[word, count] : sentence)
                    triplets.push_back(Triplet<double>(i, vocabulary[word], count));
            }
        }
        spm.setFromTriplets(triplets.begin(), triplets.end());
        spm.makeCompressed();
        spm.finalize();
        return spm;
    }

    VectorXd  inverse_document_frequency(const SparseMatrix<double>& tf_new){
        size_t n_cols = tf_new.cols();
        VectorXd idf_new(n_cols); // TODO: SparseVector?
        idf_new.setZero();
        #pragma omp parallel for schedule(guided) num_threads(8)
        for(size_t i=0; i<n_cols; i++){
            int cnt = 0;
            for(SparseMatrix<double>::InnerIterator it(tf_new, i); it; ++it){
                if(it.value() != 0)
                    cnt++;
            }
            idf_new.coeffRef(i) = log((double)(1+tf_new.rows())/(1+cnt)) + 1;
        }
        return idf_new.transpose();
    }

    unordered_map<string, double> term_frequency(const vector<string>& sentence){
        // tf(w, d) = f(w, d)
        // f(w, d) = # appears / words document
        unordered_map<string, double> words_tf;
        words_tf.reserve(sentence.size()); // Pre-allocate memory
        for(const auto& w : sentence)
            ++words_tf[w];
        const double norm = 1.0 / sentence.size(); // Compute normalization value outside the loop
        for(auto& w : words_tf)
            w.second *= norm;
        return words_tf;
    }

    SparseMatrix<double> term_freq_inverse_doc_freq(SparseMatrix<double> tf_spm, const VectorXd& idf_vxd){
        #pragma omp parallel for schedule(guided) num_threads(8)
        {
            for(size_t i=0; i<tf_spm.cols(); ++i){
                for(SparseMatrix<double>::InnerIterator it(tf_spm, i); it; ++it){
                    it.valueRef() *= idf_vxd.coeffRef(i); 
                }
            }
        }
        return tf_spm;
    }

    public:

    unordered_map<string, size_t> vocabulary;
    map<size_t, string> reverse_vocabulary;
    VectorXd _idf;


    TFIDF(string stopwords=""){
        sw = stopwords;
    }
    
    template <typename T>

    void visualize_sparse_matrix(SparseMatrix<T> ex_spm){
        cout << "Matrix (" << ex_spm.rows() <<"x"<< ex_spm.cols() << ")" << endl;
        for(size_t i=0; i<ex_spm.rows(); ++i)
            cout << ex_spm.row(i);
    }

    TFIDF& fit(const vector<string>& raw_document){
        // TODO: Only validations
        return *this;
    }

    shared_ptr<SparseMatrix<double>> transform(const vector<string>& raw_document){
        Tokenizer tokenizer = Tokenizer(sw);
        auto iniT = chrono::high_resolution_clock::now();
        auto tokens = tokenizer.fit_transform(raw_document);
        auto endT = chrono::high_resolution_clock::now();
        auto exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by Tokenizer: " << exec_time.count() << " seconds" << endl;
        set<string> vocab_keys;
        for(const auto& tok : tokens){
            vocab_keys.insert(tok.begin(), tok.end());
        }
        iniT = chrono::high_resolution_clock::now();
        //thread t_create_map(&TFIDF::create_map_vocabulary, this, cref(vocab_keys));
        create_map_vocabulary(vocab_keys);
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by create_map_vocabulary sequential: " << exec_time.count() << " seconds" << endl;

        vector<unordered_map<string, double>> aux_tf;
        aux_tf.reserve(tokens.size());
        iniT = chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(guided) num_threads(8)
        {
            for (size_t i=0; i < tokens.size(); i++) {
                const auto& sentence = tokens.at(i);
                aux_tf.push_back(term_frequency(sentence));
            }
        }
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by TF old guided: " << exec_time.count() << " seconds" << endl;
        this->tf = aux_tf;

        iniT = chrono::high_resolution_clock::now();
        SparseMatrix<double> tf_spm = convert_to_sparsematrix(vocab_keys.size());
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by Convert to SparseMatrix: " << exec_time.count() << " seconds" << endl;


        iniT = chrono::high_resolution_clock::now();
        this->_idf = inverse_document_frequency(tf_spm);
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by idf: " << exec_time.count() << " seconds" << endl;

        iniT = chrono::high_resolution_clock::now();
        this->_tf_idf_spm = term_freq_inverse_doc_freq(tf_spm, this->_idf);
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by transform: " << exec_time.count() << " seconds" << endl;
        this->_tf_idf_spm.makeCompressed();
        this->_tf_idf_spm.finalize();
        return make_shared<SparseMatrix<double>>(this->_tf_idf_spm); //TODO: Problem returning huge objects, take a lot of time
    }

    SparseMatrix<double> fit_transform(const vector<string>& raw_document){
        fit(raw_document);
        auto result = transform(raw_document);
        return *result;
    }
};