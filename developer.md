# Developer notes

## TODOs
- TODO: Optimize tokenizer
- TODO: Optimize return SparseMatrix
- TODO: Optimize set<string> unique words
- TODO: Apply max_dt & min_df
- TODO: Well format, using OOP?, virtual?, ...?
- TODO: Make something to be able to execute it, like cmake or whatever

## Run time

Tests done using test_performance.cpp.

- Tokenizer time: 47 seconds
- unque keys time: 56 seconds
- Create map time: 11 seconds
- Generate tf time: 10 seconds
- Move tf time: 0 seconds
- convert to sparsematrix time: 8 seconds
- Total time: 150 seconds

## Interesting things

It is interesting to analyse which option gives better results.
```
    #pragma omp parallel for schedule(dynamic, 100) num_threads(8)
    #pragma omp parallel for schedule(guided) num_threads(16)
```
