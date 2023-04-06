# Mondongo NLP

Trying to be scikit-learn but worse.

## Implementation of:

- TF-IDF
- Tokenizer
- Sparse matrix

## Languages supported:

- English

## Performance comparison:

### Specifications

| Spec.     | Value                                                 |
| --------- | -------------                                         |
| RAM       |  16 GB                                                |
| Processor | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz   2.50 GHz  |

### Code

```
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('data.csv', sep=';')
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['text'].values)
```

```
/* READ DATA */

TFIDF tfidf = TFIDF("en");
auto x = tfidf.fit_transform(corpus_file_b);
```

### Performance Results

Shape test data: 393.736 documents

| Spec.        | Exec. time |
| ---------    | ------     |
| scikit-learn | 30s        |
| mondongo_nlp | 4 min      |
