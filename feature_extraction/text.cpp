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

#include "..\sparse_matrix.cpp"

using namespace std;

// TODOs:
// TODO: make something to be able to execute it,, like cmake or whatever
// TODO: redefine TF calculation, somehow normalize
// TODO: check what happens when one word in doc
// TODO: toarray() function
// TODO: transform function
// TODO: fit_transform function
// TODO: Compare results with python code (Speed and Memory print)
// TODO: put everything in a well format, using OOP, virtual, ...


class Tokenizer{
    private:
    vector<string> en_sw;
    vector<string> valid_lang = {"en"};

    void load_stopwords(string lang){
        ifstream en_stopwords; en_stopwords.open("../stopwords_"+lang+".txt");
        if(en_stopwords){
            string line;
            while(getline(en_stopwords, line)){
                en_sw.push_back(line);
            }
        }
        else{
            cerr << "Could not load English stopwords";
        }
    }

    public:
    Tokenizer(string lang="en"){
        assert(find(valid_lang.begin(), valid_lang.end(), lang) != valid_lang.end() && "Stopwords requested are not supported");
        load_stopwords(lang);
    }

    vector<vector<string>> fit_transform(vector<string> &sentences){
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

    void create_map_vocabulary(set<string> vocab_keys){
        size_t i = 0;
        for(string word : vocab_keys){
            this->vocabulary[word] = i;
            i++;
        }        
    }

    unordered_map<string, double> TF(vector<string> sentence){
        // tf(w, d) = f(w, d)
        // f(w, d) = # appears / words document
        unordered_map<string, double> words_tf;
        for(int i=0; i<sentence.size(); i++)
            words_tf[sentence[i]]++;
        for(auto w : words_tf)
            words_tf[w.first] = w.second / sentence.size();
        return words_tf;
    }
    unordered_map<string, double>  IDF(vector<vector<string>> tokens, set<string> words){
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

    map<string, size_t> vocabulary;
    csr_matrix csr_tfidf;

    TFIDF(string stopwords=""){
        sw = stopwords;
    }

    void fit(vector<string> raw_document){
        Tokenizer tokenizer = Tokenizer(sw);
        auto tokens = tokenizer.fit_transform(raw_document);
        vector<unordered_map<string, double>> tf;
        set<string> vocab_keys;
        for(auto tok : tokens){
            auto aux = TF(tok);
            tf.push_back(aux);
            vocab_keys.insert(tok.begin(), tok.end());
        }
        auto idf = IDF(tokens, vocab_keys);
        create_map_vocabulary(vocab_keys);
        vector<size_t> indptr = {0};
        vector<size_t> indices;
        vector<double> data;
        for(size_t d=0; d<tokens.size(); d++){
            for(auto w : tf[d]){
                indices.push_back(vocabulary[w.first]);
                data.push_back(w.second * idf[w.first]);
            }
            indptr.push_back(indices.size());
        }
        csr_tfidf = csr_matrix(indptr, indices, data);
    }
};
