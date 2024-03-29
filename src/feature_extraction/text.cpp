#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <execution>
#include <omp.h>

// #include <cuda.h> future? download it
#include "..\..\third_party\eigen3\Eigen\Sparse"
#include "..\..\third_party\eigen3\Eigen\SparseCore"
#include "..\..\third_party\eigen3\Eigen\Dense"

#include ".\stop_words.cpp"

using namespace Eigen;
using namespace std;



class BaseEstimatorText{
    private:
    virtual unordered_map<string, double> term_frequency(const vector<string>& sentence) = 0;

    protected:
    unordered_map<string, size_t> vocabulary;
    map<size_t, string> reverse_vocabulary;
    vector<unordered_map<string, double>> tf;

    public:
    virtual shared_ptr<SparseMatrix<double>> transform(const vector<string>& raw_document) = 0;
    virtual shared_ptr<SparseMatrix<double>> fit_transform(const vector<string>& raw_document) = 0;
    void create_map_vocabulary(const set<string>& vocab_keys){
        size_t i = 0;
        for(const auto& word : vocab_keys){
            vocabulary[word] = i;
            reverse_vocabulary[i] = word;
            i++;
        }
    }
    void visualize_sparse_matrix(SparseMatrix<double>& ex_spm){
        cout << "Matrix (" << ex_spm.rows() <<"x"<< ex_spm.cols() << ")" << endl;
        for(size_t i=0; i<ex_spm.rows(); ++i)
            cout << ex_spm.row(i);
    }
    
    SparseMatrix<double> convert_to_sparsematrix(size_t col_dim) {
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
};

class Tokenizer{
    private:
    unordered_set<string> en_sw;
    const vector<string> valid_lang = {"en"};

    void load_stopwords(string lang, const unordered_set<string> &sw_custom){
        if(sw_custom.empty())
            en_sw = *sw_lang_map[lang];
        else
            en_sw = sw_custom;
    }

    vector<string> preprocess_sentence(const vector<string>& sentences){
        vector<string> preprocessed_sentences(sentences.size());

        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
        for(size_t i = 0; i < sentences.size(); i++){
            string lowercase_line(sentences[i]);
            lowercase_line.erase(remove_if(lowercase_line.begin(), lowercase_line.end(), ::ispunct), lowercase_line.end());
            transform(lowercase_line.begin(), lowercase_line.end(), lowercase_line.begin(), ::tolower);
            preprocessed_sentences[i] = move(lowercase_line);
        }
        return preprocessed_sentences;
    }

    public:
    Tokenizer(const string& lang="en", const unordered_set<string>& sw_custom = {}){
        if(find(valid_lang.begin(), valid_lang.end(), lang) == valid_lang.end() && sw_custom.empty()){
            cerr << "ERROR: Stopwords requested are not supported. Add custom stop words or use one of the predefined ones" << endl;
            throw new exception();
        }
        load_stopwords(lang, sw_custom);
    }
    
    vector<vector<string>> fit_transform(const vector<string>& sentences){
        vector<vector<string>> tokens;
        tokens.reserve(sentences.size());
        
        auto preprocessed_sentences = preprocess_sentence(sentences);

        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
        for(const auto& line : preprocessed_sentences){
            istringstream is(line);
            string aux;
            vector<string> tok;
            while(getline(is, aux, ' ')){
                if(en_sw.count(aux) == 0){
                    tok.emplace_back(move(aux));
                }
            }
            tokens.emplace_back(move(tok));
        }
        return tokens;
    }

    void visualize_tokens(vector<vector<string>> &tokens) const{
        for(const auto& i : tokens){
            for(const auto& j : i){
                cout << j + " ";
            }
            cout << "\n";
        }
    }
};


class TFIDF : public BaseEstimatorText {
    private:
    string sw_lang;
    //vector<unordered_map<string, double>> tf;
    SparseMatrix<double> _tf_idf_spm;
    double max_df = 1.0, min_df = 0.0;

    VectorXd  inverse_document_frequency(const SparseMatrix<double>& tf_new){
        size_t n_cols = tf_new.cols();
        VectorXd idf_new(n_cols); // TODO: SparseVector?
        idf_new.setZero();
        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
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

    unordered_map<string, double> term_frequency(const vector<string>& sentence) override {
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
        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
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
    VectorXd _idf;
    unordered_set<string> sw_custom;

    TFIDF(string sw_lang="", unordered_set<string> sw_custom={}, double max_df=1.0, double min_df=0.0){
        this->sw_lang = sw_lang;
        this->max_df = max_df;
        this->min_df = min_df;
        this->sw_custom = sw_custom;
    }

    unordered_map<string, size_t> get_vocabulary(){ return vocabulary; }

    TFIDF& fit(const vector<string>& raw_document) {
        // TODO: Only validations
        if(this->max_df > 1.0 || this->max_df < 0.0) // If out of the range then default
            this->max_df = 1.0;
        if(this->max_df < this->min_df || this->min_df < 0.0)
            min_df = 0;
        return *this;
    }

    shared_ptr<SparseMatrix<double>> transform(const vector<string>& raw_document) override {
        Tokenizer tokenizer = Tokenizer(this->sw_lang, this->sw_custom);
        auto tokens = tokenizer.fit_transform(raw_document);
        set<string> vocab_keys;
        for(const auto& tok : tokens){
            vocab_keys.insert(tok.begin(), tok.end());
        }
        create_map_vocabulary(vocab_keys);

        vector<unordered_map<string, double>> aux_tf;
        aux_tf.reserve(tokens.size());
        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
        {
            for (size_t i=0; i < tokens.size(); i++) {
                const auto& sentence = tokens.at(i);
                aux_tf.push_back(term_frequency(sentence));
            }
        }

        this->tf = move(aux_tf);

        SparseMatrix<double> tf_spm = convert_to_sparsematrix(vocab_keys.size());

        this->_idf = inverse_document_frequency(tf_spm);

        this->_tf_idf_spm = term_freq_inverse_doc_freq(tf_spm, this->_idf);
        
        this->_tf_idf_spm.prune([&](int row, int col, double value){return value > 0.001; }); // Prune values smaller than 0.001
        this->_tf_idf_spm = this->_tf_idf_spm.unaryExpr([](double value) -> double {
            if(value > 1.0)    
                return 1.0;
            else
                return value;
        });
        this->_tf_idf_spm.makeCompressed();
        this->_tf_idf_spm.finalize();

        return make_shared<SparseMatrix<double>>(move(this->_tf_idf_spm)); //TODO: Problem returning huge objects, take a lot of time
    }

    shared_ptr<SparseMatrix<double>> fit_transform(const vector<string>& raw_document) override {
        fit(raw_document);
        return transform(raw_document);
    }
};


class CountVectorizer : public BaseEstimatorText{
    private:
    string sw_lang;
    double max_df=1.0, min_df=0.0;

    public:
    unordered_set<string> sw_custom;

    CountVectorizer(string sw_lang="", unordered_set<string> sw_custom={}, double max_df=1.0, double min_df=0.0){
        this->sw_lang = sw_lang;
        this->max_df = max_df;
        this->min_df = min_df;
        this->sw_custom = sw_custom;
    }

    CountVectorizer& fit(const vector<string>& raw_document){
        if(this->max_df > 1.0 || this->max_df < 0.0) // If out of the range then default
            this->max_df = 1.0;
        if(this->max_df < this->min_df || this->min_df < 0.0)
            min_df = 0;
        return *this;
    }

    unordered_map<string, size_t> get_vocabulary(){ return vocabulary; }

    unordered_map<string, double> term_frequency(const vector<string>& sentence) override {
        unordered_map<string, double> words_tf;
        words_tf.reserve(sentence.size()); // Pre-allocate memory
        for(const auto& w : sentence)
            ++words_tf[w];
        return words_tf;
    }

    shared_ptr<SparseMatrix<double>> transform(const vector<string>& raw_document) override {
        Tokenizer tokenizer = Tokenizer(this->sw_lang, this->sw_custom);
        auto tokens = tokenizer.fit_transform(raw_document);
        set<string> vocab_keys;
        for(const auto& tok : tokens){
            vocab_keys.insert(tok.begin(), tok.end());
        }
        create_map_vocabulary(vocab_keys);

        vector<unordered_map<string, double>> aux_tf;
        aux_tf.reserve(tokens.size());
        #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
        {
            for (size_t i=0; i < tokens.size(); i++) {
                const auto& sentence = tokens.at(i);
                aux_tf.push_back(term_frequency(sentence));
            }
        }

        this->tf = move(aux_tf);

        SparseMatrix<double> tf_spm = convert_to_sparsematrix(vocab_keys.size());

        return make_shared<SparseMatrix<double>>(move(tf_spm));
    }

    shared_ptr<SparseMatrix<double>> fit_transform(const vector<string>& raw_document) override {
        fit(raw_document);
        return transform(raw_document);
    }

};