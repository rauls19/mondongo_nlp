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
#include <execution>
#include <chrono>
#include <thread>
#include <mutex>

#include "..\sparse_matrix.cpp"

using namespace std;

// TODOs:
// TODO: parallelism exection?
// TODO: check what happens when one word in doc
// TODO: toarray() function
// TODO: pointers etc
// TODO: put everything in a well format, using OOP?, virtual?, ...?
// TODO: make something to be able to execute it, like cmake or whatever
// TODO: redefine TF calculation, somehow normalize


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


    void create_map_vocabulary(const set<string>& vocab_keys){
        size_t i = 0;
        for(const auto& word : vocab_keys){
            vocabulary[word] = i;
            i++;
        }
    }

    void TF(vector<string> sentence){
        // tf(w, d) = f(w, d)
        // f(w, d) = # appears / words document
        //Segmentation error somewhere
        unordered_map<string, double> words_tf;
        words_tf.reserve(sentence.size()); // Pre-allocate memory
        for(const auto& w : sentence)
            words_tf[w]++;
        const double norm = 1.0 / sentence.size(); // Compute normalization value outside the loop
        for(auto& w : words_tf)
            w.second *= norm;
        tf.push_back(move(words_tf)); // Use move semantics
    }

    unordered_map<string, double> IDF(vector<vector<string>> tokens, const set<string>& words){
        // idf(w, D) = log(N/f(w, D))
        // n = 4 # number of documents
        // df_t = 1
        // idf = math.log((1 + n) / (1 + df_t)) + 1
        unordered_map<string, double> words_idf;
        size_t n_docs = tokens.size();
        // could be done parallel
        for(const auto& sentence : tokens){
            set<string> unique_words(sentence.begin(), sentence.end());
            for(auto& word : unique_words)
                words_idf[word]++;
        }
        // could be done parallel
        for(const auto& word : words)
            words_idf[word] = log((1+n_docs) / (1+words_idf[word])) + 1;
        return words_idf;
    }

    public:

    map<string, size_t> vocabulary;
    csr_matrix csr_tfidf;
    unordered_map<string, double> _idf;


    TFIDF(string stopwords=""){
        sw = stopwords;
    }
    
    TFIDF& fit(const vector<string>& raw_document){
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
        vector<thread> threads;
        threads.reserve(tokens.size());
        for(const auto& tok : tokens) {
            threads.emplace_back(&TFIDF::TF, this, move(tok));
        }
        for(auto& t : threads){
            t.join();
        }
        //t_create_map.join();
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by TF sequential: " << exec_time.count() << " seconds" << endl;
        
        iniT = chrono::high_resolution_clock::now();
        _idf = IDF(tokens, vocab_keys); // time consuming
        endT = chrono::high_resolution_clock::now();
        exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
        cout << "Time taken by IDF: " << exec_time.count() << " seconds" << endl;

        return *this;
    }

    csr_matrix transform(const vector<string>& raw_document){
        vector<size_t> indptr = {0};
        vector<size_t> indices;
        vector<double> data;
        for(size_t d=0; d<tf.size(); d++){
            for(auto w : tf.at(d)){
                indices.push_back(vocabulary[w.first]);
                if (w.second == 1)
                    w.second = w.second / _idf[w.first];
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