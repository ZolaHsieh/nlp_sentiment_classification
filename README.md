# NLP Sentiment Classification
LSTM, LSTM with glove-pretrain word vec, Bidirectional LSTM, DistilBert Fine-tuning sentiment analysis in PyTorch

## Dataset: IMDb
* Download directly: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
* HuggingFace dataset: https://huggingface.co/datasets/imdb
```python
from datasets import load_dataset
load_dataset('imdb').with_format('torch') # for pytorch format
```

## Glove
Apply pre-train glove pre-train word vector for LSTM embedding initialization
* DataSource
    * github: https://github.com/stanfordnlp/GloVe
    * Kaggle: https://www.kaggle.com/datasets/anindya2906/glove6b
* Preprocess: transform gloveweight to gensim format
```bash
unzip glove.6B.zip
python -m gensim.scripts.glove2word2vec --input glove.6B.100d.txt --output glove.6B.100d.w2vformat.txt
```
```python
## load glove pre-train weight
import gensim
word_vec = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.w2vformat.txt', binary=False, encoding='utf-8')
```
## Performance on Training & Test set (fo ref)
### Loss
|model|training loss|test loss|
|---|---|---|
|LSTM|0.1315|0.4560|
|LSTM+GLOVE|0.1501|0.5397|
|Bidirectional LSTM|0.1251|0.4253|
|Fine-tuning DistilBert|0.3820|0.3415|

### Accuracy
|model|training accuracy (%) |test accuracy (%)|
|---|---|---|
|LSTM|95.73|84.54|
|LSTM+GLOVE|94.88|82.05|
|Bidirectional LSTM|95.96|84.39|
|Fine-tuning DistilBert|83.10|85.10|

## Demo
Use [gradio](https://www.gradio.app/) to demo model result, please click [link](https://huggingface.co/spaces/zolakarary/SentimentClf) to give it a try!
![image](demo.png)

