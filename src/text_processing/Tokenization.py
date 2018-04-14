from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.tokenize import PunktSentenceTokenizer


def tokenize_sentence(sentences):
    sent_tokens = sent_tokenize(sentences)
    return sent_tokens


def tokenize_words(text):
    word_tokens = word_tokenize(text)
    return word_tokens


def tokenize_words_without_stopwords(text, custom_stopwords):
    word_tokens = word_tokenize(text)
    stop_words_list = set.union(set(stopwords.words("english")), set(custom_stopwords))
    filtered_sentence = [w for w in word_tokens if not w in stop_words_list]
    stop_words = [w for w in word_tokens if w in stop_words_list]
    return filtered_sentence, stop_words


def stem_tokens(text):
    word_tokens = tokenize_words(text)
    ps = PorterStemmer()
    words_tokens_after_stemming = [ps.stem(w) for w in word_tokens]
    return words_tokens_after_stemming


def tag_pos(word):
    token = tokenize_words(word)
    return pos_tag(token)







