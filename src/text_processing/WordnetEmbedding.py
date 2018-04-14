from nltk.corpus import wordnet


def get_synonym(word):
    synonym = wordnet.synsets(word)
    print(synonym)
    return synonym