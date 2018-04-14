import collections

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

from src.text_processing import Tokenization


class TextProcessor:

    def __init__(self):
        self._tokenizer = Tokenizer(filters='!"#$%&./:?@[\\]`~\t\n', lower=False)

    def train_tokenizer(self, data):
        self._tokenizer.fit_on_texts(data)

    def convert_text_to_matrix(self, data, mode="binary"):
        one_hot_encoded_docs = self._tokenizer.texts_to_matrix(data, mode=mode)
        return one_hot_encoded_docs

    def convert_text_to_sequences(self, texts):
        sequences = self._tokenizer.texts_to_sequences(texts)
        return sequences

    def get_word_index(self, word):
        index = self._tokenizer.word_index.get(word)
        return index

    def get_word_index_keys(self):
        return self._tokenizer.word_index.keys()

    def get_word_index_values(self):
        return self._tokenizer.word_index.values()

    @staticmethod
    def _read_words(sentences):
        word_list = []
        word_count = 0
        for i in range(len(sentences)):
            word_count += len(sentences[i])
            words = Tokenization.tokenize_words(sentences[i])
            word_list.extend(words)
        return word_list, word_count

    @staticmethod
    def build_vocab(sentences):
        word_list, word_count = TextProcessor._read_words(sentences)
        counter = collections.Counter(word_list)
        word_count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = dict((i, c) for i, c in enumerate(words))
        return word_to_id, id_to_word

    # Return words with frequency count equals one
    @staticmethod
    def get_stopwords(sentences):
        word_list, word_count = TextProcessor._read_words(sentences)
        counter = collections.Counter(word_list)
        stopwords = [word for word, frequency in counter.items() if frequency <= 3]
        return stopwords

    @staticmethod
    def _file_to_word_ids(sentences, word_to_id):
        word_list, _ = TextProcessor._read_words(sentences)
        return [word_to_id[word] for word in word_list if word in word_to_id]

    @staticmethod
    def process_data(data, stopwords):
        train_data = np.array(data).reshape(len(data))
        # Remove Pos and Neg words from every row
        text = ''
        for i in range(len(data)):
            filtered_sentence, stop_words = Tokenization.tokenize_words_without_stopwords(train_data[i], stopwords)
            stemmed_sent = text_to_word_sequence(' '.join(Tokenization.stem_tokens(' '.join(filtered_sentence))))
            train_data[i] = ' '.join(stemmed_sent)
            text += train_data[i]
        return train_data

    @staticmethod
    def process_text(filename):
        data = open(filename, 'r').read().replace('\n', '')
        entire_text = np.array(data).reshape(1)

        for i in range(len(entire_text)):
            stemmed_sent = ' '.join(text_to_word_sequence(' '.join(Tokenization.stem_tokens(entire_text[i]))))
            filtered_sentence, stop_words = Tokenization.tokenize_words_without_stopwords(stemmed_sent)
            entire_text[i] = ' '.join(filtered_sentence)
        return entire_text

    @staticmethod
    def process_input_data(sentences):
        processed_data = []
        for sent in sentences:
            word_and_label_list = sent[0].splitlines()
            input_sent = []
            for word_and_label in word_and_label_list:
                word = word_and_label.split(' ')[0]
                label = word_and_label.split(' ')[1]
                pos_word = Tokenization.tag_pos(word)[0][1]
                input_sent.append((word, pos_word, label))
                # wordnet_synonym = WordnetEmbedding.get_synonym(word)
                # output_file.write(word + ' ' + pos_word + ' ' + label + '\n')
            # output_file.write('\n')
            processed_data.append(input_sent)
        return processed_data







