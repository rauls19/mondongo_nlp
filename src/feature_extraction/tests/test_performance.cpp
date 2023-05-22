#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

#include "..\text.cpp"

using namespace std;

int main(){
    vector<string> corpus_file_b;
    ifstream file_data_b; file_data_b.open("description_b.csv");
    if(file_data_b){
        string line;
        while(getline(file_data_b, line)){
            corpus_file_b.push_back(line);
        }
    }
    else{
        cerr << "Could not load Descriptions file" << endl;
    }
    
    TFIDF tfidf = TFIDF("en");
    auto iniT = chrono::high_resolution_clock::now();
    auto x = tfidf.fit_transform(corpus_file_b);
    auto endT = chrono::high_resolution_clock::now();
    auto exec_time = chrono::duration_cast<chrono::seconds>(endT - iniT);
    cout << "Time taken by function: " << exec_time.count() << " seconds" << endl;
    return 0;
    // 6min process 393.736 sentences vs 20 secs sklearn
}

/*

Time taken by Tokenizer: 84 seconds
Time taken by TF parallel: 56 seconds
Time taken by IDF: 85 seconds
Time taken by create_map_vocabulary: 70 seconds
~30s build matrix
Time taken by function: 330 seconds

*/