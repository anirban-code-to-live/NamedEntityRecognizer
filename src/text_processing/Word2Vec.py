from gensim.models import word2vec as w2v
from gensim.models import KeyedVectors as kv
import os


def generate_embedding(sentences):
    _model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models')
    _model = w2v.Word2Vec(sentences, size=3)
    _model.wv.save_word2vec_format(os.path.join(_model_path, 'ner.model.bin'), binary=True)


def get_embedding(word):
    _model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models')
    _model = kv.load_word2vec_format(os.path.join(_model_path, 'ner.model.bin'), binary=True)
    try:
        embedding = _model.wv[word]
    except KeyError:
        embedding = None
    return embedding


def get_most_similar_word(word):
    _model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models')
    _model = kv.load_word2vec_format(os.path.join(_model_path, 'ner.model.bin'), binary=True)
    try:
        return _model.most_similar([word], topn=1)[0][0]
    except KeyError:
        return None