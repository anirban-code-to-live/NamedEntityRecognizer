def word2features(sent, i, pos_tag, title_feature, upper_case_feature, embed_feature, similar_feature):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower()
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower()
        })
    else:
        features['EOS'] = True

    # Handling of POS_TAG feature
    if pos_tag:
        __add_pos_tag(sent, i, features)

    # Handling of Uppercase character feature
    if upper_case_feature:
        __add_uppercase_feature(sent, i, features)

    # Handling of title feature
    if title_feature:
        __add_title_feature(sent, i, features)

    # Handle embedding feature
    if embed_feature:
        __add_word_embedding_feature(sent, i, features)

    # Handle similar word feature
    if similar_feature:
        __add_most_similar_word_feature(sent, i, features)

    return features


def __add_pos_tag(sent, i, features):
    pos_tag = sent[i][1]

    features.update({
        'postag': pos_tag,
        'postag[:2]': pos_tag[:2]
    })

    if i > 0:
        postag_prev_word = sent[i - 1][1]
        features.update({
            '-1:postag': postag_prev_word,
            '-1:postag[:2]': postag_prev_word[:2],
        })

    if i < len(sent) - 1:
        postag_next_word = sent[i + 1][1]
        features.update({
            '+1:postag': postag_next_word,
            '+1:postag[:2]': postag_next_word[:2],
        })


def __add_uppercase_feature(sent, i, features):
    word = sent[i][0]

    features.update({
        'word.isupper()': word.isupper()
    })

    if i > 0:
        word_prev = sent[i - 1][0]
        features.update({
            '-1:word.isupper()': word_prev.isupper()
        })

    if i < len(sent) - 1:
        word_next = sent[i + 1][0]
        features.update({
            '+1:word.isupper()': word_next.isupper()
        })


def __add_title_feature(sent, i,  features):
    word = sent[i][0]
    features.update({
        'word.istitle()': word.istitle()
    })

    if i > 0:
        word_prev = sent[i - 1][0]
        features.update({
            '-1:word.istitle()': word_prev.istitle(),
        })

    if i < len(sent) - 1:
        word_next = sent[i + 1][0]
        features.update({
            '+1:word.istitle()': word_next.istitle(),
        })


def __add_most_similar_word_feature(sent, i, features):
    similar_word = sent[i][4]
    if similar_word is not None:
        features.update({
            'word.most_similar': similar_word
        })


def __add_word_embedding_feature(sent, i, features):
    word_embedding = sent[i][3]
    if word_embedding is not None:
        features.update({
            'word_embedding' : word_embedding
        })


def sent2features(sent, pos_tag=False, title_feature=False, upper_case_feature=False, embed=False, similar=False):
    return [word2features(sent, i, pos_tag, title_feature, upper_case_feature, embed, similar) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, pos_tag, label, _, _ in sent]


def sent2tokens(sent):
    return [token for token, pos_tag, label, _, _ in sent]