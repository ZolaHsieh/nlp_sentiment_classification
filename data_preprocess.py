import re
import torch
import gensim
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def load_imdb_data():
   dataset = load_dataset("imdb").with_format("torch") 
   return dataset['train'], dataset['test']


TAG_RE = re.compile(r'<[^>]+>')
def preprocess_text(sen):

    # Removing html tags
    sentence = TAG_RE.sub('', sen['text'])
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sen['text'] = sentence
    return sen


def get_tknr():
   return get_tokenizer("basic_english")


def build_vocabulary(tokenizer, datasets):
  for data in datasets:
    yield tokenizer(data['text'])


def create_vocab(tokenizer, data):
    vocab = build_vocab_from_iterator(build_vocabulary(tokenizer, data), min_freq=1)
    vocab.insert_token('<unk>', 0)
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def word2id_padding(vocab, tokenizer, data_set, max_len=128):
    X = []
    Y = []
    for data in data_set:
        x = vocab(tokenizer(data['text']))
        while len(x) < max_len:
            x.append(0)

        X.append(x if len(x) < max_len else x[:max_len])
        Y.append(data['label'])
    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)


def get_glove_weight(vocab, emb_dim):
    # use glove.6B.100d and the emb_dim = 100
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.w2vformat.txt', 
                                                              binary=False, 
                                                              encoding='utf-8')
    
	## map golve pretrain weight to pytorch embedding pretrain weight
    weight = torch.zeros(len(vocab)+1, emb_dim) # given 0 if the word is not in glove
    for i in range(len(wvmodel.index_to_key)):
        try:
            index = vocab[wvmodel.index_to_key[i]] #transfer to our word2ind
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))  
    return weight