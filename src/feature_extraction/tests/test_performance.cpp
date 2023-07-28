#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "..\text.cpp"

using namespace std;

int main(){
    vector<string> corpus_file_b;
    ifstream file_data_b; file_data_b.open("description_b.csv");
    if(file_data_b){
        string line;
        while(getline(file_data_b, line))
            corpus_file_b.push_back(line);
    }
    else{
        cerr << "Could not load Descriptions file" << endl;
    }

    TFIDF tfidf = TFIDF("en");
    auto iniT = chrono::high_resolution_clock::now();
    auto x = *tfidf.fit_transform(corpus_file_b);
    auto endT = chrono::high_resolution_clock::now();
    auto exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
    cout << "TFIDF Total time: " << exec_time.count() << " seconds" << endl;

    CountVectorizer cntvec = CountVectorizer("en");
    iniT = chrono::high_resolution_clock::now();
    auto y = *cntvec.fit_transform(corpus_file_b);
    endT = chrono::high_resolution_clock::now();
    exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
    cout << "CountVectorizer Total time: " << exec_time.count() << " seconds" << endl;
    return 0;
}