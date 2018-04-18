import os
import numpy as np
from src.text_processing import Tokenization
from src.text_processing.TextPreprocessor import TextProcessor as tp
from src.text_processing import WordnetEmbedding
from src.text_processing import FeatureManager as fm
from src.generative_models.CRFClassifier import CRFClassifier as crf
from src.generative_models.HMMClassifier import HMMClassifier as hmm
from sklearn.model_selection import KFold

if __name__ == '__main__':

    # Create directory and file for output data save
    project_relative_path = os.path.dirname(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(project_relative_path, "saved_models")):
        os.makedirs(os.path.join(project_relative_path, "saved_models"))
    if not os.path.exists(os.path.join(project_relative_path, "output_data")):
        os.makedirs(os.path.join(project_relative_path, "output_data"))
    saved_model_path = os.path.join(project_relative_path, 'saved_models')
    project_data_path = os.path.join(project_relative_path, 'dataset')
    print('Project directory :: ' + str(project_relative_path))

    input_file = open(os.path.join(project_data_path, 'ner.txt'), 'r', encoding='latin-1')
    output_file = open(os.path.join(project_data_path, 'ner_pos.txt'), 'a')
    input_sentences = input_file.read().split('\n\n')

    tp.get_data_distribution(input_sentences)

    input_sentences = np.array(input_sentences).reshape((len(input_sentences), 1))

    tp.save_sentences_without_label(input_sentences, os.path.join(project_data_path, 'ner_sentences.txt'))

    # Test purpose :: small data-set
    # input_sentences = input_sentences[0:100, :]

    kf = KFold(n_splits=10, shuffle=True, random_state=2)  # Hyper-parameters : shuffle in KFold
    for train_index, test_index in kf.split(input_sentences):
        # print('Train : ', train_index, 'Test : ', test_index)
        train_data, test_data = input_sentences[train_index], input_sentences[test_index]
        tp.generate_word2vec_embeddings(train_data)
        processed_train_data = tp.process_input_data(train_data)
        processed_test_data = tp.process_input_data(test_data)

        x_train = [fm.sent2features(sent, pos_tag=False,
                                    title_feature=False,
                                    upper_case_feature=False,
                                    embed=False,
                                    similar=False) for sent in processed_train_data]
        y_train = [fm.sent2labels(sent) for sent in processed_train_data]

        x_test = [fm.sent2features(sent, pos_tag=False,
                                   title_feature=False,
                                   upper_case_feature=False,
                                   embed=False,
                                   similar=False) for sent in processed_test_data]
        y_test = [fm.sent2labels(sent) for sent in processed_test_data]

        crf_model = crf()
        crf_model.train(x_train, y_train)

        f1_score = crf_model.get_class_f1_score(x_test, y_test)
        print(f1_score)

        crf_model.print_top_negative_features()
        crf_model.print_top_positive_features()

    input_file.close()


