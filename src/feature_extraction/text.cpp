#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include <set>
#include <map>
#include <utility>

#include "..\sparse_matrix.cpp"

using namespace std;

// TODOs:
// TODO: fit_transform function
// TODO: transform function
// TODO: toarray() function
// TODO: pointers etc
// TODO: put everything in a well format, using OOP?, virtual?, ...?
// TODO: parallelism exection
// TODO: check what happens when one word in doc
// TODO: find huge text and compare results with python
// TODO: make something to be able to execute it, like cmake or whatever
// TODO: redefine TF calculation, somehow normalize
// TODO: Compare results with python code (Speed and Memory print)


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
        for(auto line : sentences){
            line.erase(remove_if(line.begin(), line.end(), ::ispunct), line.end());
            transform(line.begin(), line.end(), line.begin(), ::tolower);
            istringstream is(line);
            string aux;
            vector<string> tok;
            while(getline(is, aux, ' ')){
                if(!(find(en_sw.begin(), en_sw.end(), aux) != en_sw.end())){
                    tok.push_back(aux);
                }
            }
            tokens.push_back(tok);
        }
        return tokens;
    }

    void visualize(vector<vector<string>> &tokens){
        for(auto i : tokens){
            for(auto j : i){
                cout << j + " ";
            }
            cout << "\n";
        }
    }
};


class TFIDF{
    private:
    string sw;
    mutable vector<unordered_map<string, double>> tf;


    void create_map_vocabulary(set<string> vocab_keys) const{
        size_t i = 0;
        for(string word : vocab_keys){
            this->vocabulary[word] = i;
            i++;
        }        
    }

    unordered_map<string, double> TF(vector<string> sentence) const{
        // tf(w, d) = f(w, d)
        // f(w, d) = # appears / words document
        unordered_map<string, double> words_tf;
        for(int i=0; i<sentence.size(); i++)
            words_tf[sentence.at(i)]++;
        for(auto w : words_tf)
            words_tf[w.first] = w.second / sentence.size();
        return words_tf;
    }
    unordered_map<string, double> IDF(vector<vector<string>> tokens, set<string> words) const{
        // idf(w, D) = log(N/f(w, D))
        // n = 4 # number of documents
        // df_t = 1
        // idf = math.log((1 + n) / (1 + df_t)) + 1
        unordered_map<string, double> words_idf;
        size_t n_docs = tokens.size();
        for(auto sentence : tokens)
            for(auto word : words)
                if(find(sentence.begin(), sentence.end(), word) != sentence.end())
                    words_idf[word]++;
        for(auto word : words)
            words_idf[word] = log((1+n_docs) / (1+words_idf[word])) + 1;
        return words_idf;
    }

    public:

    mutable map<string, size_t> vocabulary;
    csr_matrix csr_tfidf;
    mutable unordered_map<string, double> _idf;


    TFIDF(string stopwords=""){
        sw = stopwords;
    }
    
    TFIDF& fit(const vector<string>& raw_document){
        return const_cast<TFIDF&>(as_const(*this).fit(raw_document));
    }

    const TFIDF& fit(const vector<string>& raw_document) const{
        Tokenizer tokenizer = Tokenizer(sw);
        auto tokens = tokenizer.fit_transform(raw_document);
        set<string> vocab_keys;
        for(auto tok : tokens){
            auto aux = TF(tok);
            tf.push_back(aux);
            vocab_keys.insert(tok.begin(), tok.end());
        }
        _idf = IDF(tokens, vocab_keys);
        create_map_vocabulary(vocab_keys);
        return *this;
    }

    csr_matrix transform(const vector<string>& raw_document){
        vector<size_t> indptr = {0};
        vector<size_t> indices;
        vector<double> data;
        for(size_t d=0; d<tf.size(); d++){
            for(auto w : tf.at(d)){
                indices.push_back(vocabulary[w.first]);
                data.push_back(w.second * _idf[w.first]);
            }
            indptr.push_back(indices.size());
        }
        return csr_matrix(indptr, indices, data);
    }

    csr_matrix fit_transform(const vector<string>& raw_document){
        return fit(raw_document).transform(raw_document);
    }
};