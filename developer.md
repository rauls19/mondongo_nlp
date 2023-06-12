It is interesting to analyse which option gives better results.
```
    #pragma omp parallel for schedule(dynamic, 100) num_threads(8)
    #pragma omp parallel for schedule(guided) num_threads(16)
```
