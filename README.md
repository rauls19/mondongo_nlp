# Mondongo NLP

Trying to be scikit-learn but worse. From papers to code.

## Implementation of:

- Tokenizer
- TF-IDF
- CountVectorizer

## Languages supported:

- English

## Performance comparison:

### Specifications

| Spec.     | Value                                                 |
| --------- | -------------                                         |
| RAM       |  16 GB                                                |
| Processor | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz   2.50 GHz  |

### Code Example

scikit-learn:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('data.csv', sep=';')
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['text'].values)
```

mondongo_nlp:

```c++
/* READ DATA */

TFIDF tfidf = TFIDF("en");
auto x = tfidf.fit_transform(corpus_file_b);
```

### Performance Results

For 393.736 documents:

| Spec.                          | Exec. time   |
| ---------                      | ------       |
| TFIDF - scikit-learn           | 30s          |
| TFIDF - mondongo_nlp           | 147s         |
| CountVectorizer - scikit-learn | 21s          |
| CountVectorizer - mondongo_nlp | 124s         |

