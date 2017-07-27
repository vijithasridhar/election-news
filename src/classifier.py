from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
import numpy as np
import pandas as pd
from phrasemachine import phrasemachine
from collections import Counter


def concat_features():
    other_features = get_vectorizations_for_all_classes()

    print("Combining all data for %s" % ng)
    all_lexical_feats = None
    labels = []

    for i, ng in enumerate(util.newsgroups):
        lexical_feats = pd.read_csv(util.lexical_features_file % ng, sep=',', encoding='utf8') \
                        .sample(n=(num_datapoints_for_model // len(util.newsgroups)), random_state=i)
        labels.append(lexical_feats.shape[0])

        if all_lexical_feats is not None:
            all_lexical_feats = pd.concat([all_lexical_feats, lexical_feats], axis=0)
        else:
            all_lexical_feats = lexical_feats

    X = pd.concat([all_lexical_feats, other_features], axis=1)

    # delete the first column because they're all the status IDs, but save that
    status_ids_order = X[util.primary_key]    
    X.drop(util.primary_key, axis=1, inplace=True)

    # kept track of just how many datapoints for each newsgroup there was
    y = np.repeat(range(1, len(util.newsgroups) + 1), labels)
    return X, y


# returns the named entities and ngrams (vectorized) into a dataframe
def get_vectorizations_for_all_classes():

    # concatenate all headlines into a 1D array so you can extract all ngrams/NEs from them
    all_headlines = None
    
    for i, ng in enumerate(util.newsgroups):

        # sample datapoints from all the headlines, since you have way too many headlines to use in a model
        all_link_data = pd.read_csv(util.datafile % ng, sep=',', encoding='utf8') \
                        .sample(n=(num_datapoints_for_model // len(util.newsgroups)), random_state=i)
        
        if all_headlines is not None:
            all_headlines = pd.concat([all_headlines, all_link_data['link_name']], axis=0)
        else:
            all_headlines = all_link_data['link_name']


    # get named entities with phrasemachine
    print("Generating named entities for all data")
    phrases = sum((phrasemachine.get_phrases(h)['counts'] for h in all_headlines), \
			Counter()).most_common(top_nes)
    vectorizer = CountVectorizer(vocabulary=[k[0] for k in all_headlines_phrases])
    ner = vectorizer.fit_transform(all_headlines)
    ner = pd.DataFrame(data=ner, columns=vectorizer.get_feature_names())

    # ngrams
    print("Generating ngrams for all data")
    vectorizer = CountVectorizer(ngram_range=(min_n, max_n), analyzer=analyzer, max_features=top_ngrams)
    ngrams = vectorizer.fit_transform(all_headlines)
    ngrams = pd.DataFrame(data=ngrams, columns=vectorizer.get_feature_names())

    return pd.concat([ner, ngrams], axis=1)
    


def train(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(   \
                    X, Y, test_size=0.2, random_state=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print "Training error: %.6f" % accuracy_score(model.predict(X_train), y_train)
    print "Test error: %.6f" % accuracy_score(y_pred, y_test)
    
    confusion_matrices_pdf.close()
    print_best_features(model.feature_importances_, X.columns)


def print_best_features(coef, feature_names):
    #zip two lists together to iterate through them simultaneously
    #sort the two lists by the values in the first (the coefficients)
    zipped = zip(coef, feature_names)				
    zipped.sort(key = lambda t: t[0], reverse=True)

    print "\nMOST POSITIVE FEATURES:"
    for (weight, word) in zipped[:40]:
        print "{}\t{:.6f}".format(word, weight)

    print "\nMOST NEGATIVE FEATURES:"
    for (weight, word) in zipped[:-40:-1]:
        print "{}\t{:.6f}".format(word, weight)


def accuracy_score(y_pred, y_true):
    # plot confusion matrix for classes
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=util.newsgroups)
    confusion_matrices_pdf.savefig(plt.gcf())
    
    #return mae(y_true, y_pred)


# taken from sklearn documentation
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    # TODO hyperparameters for ngrams/NER
    min_n = 1
    max_n = 2
    analyzer = 'word'
    top_ngrams = 200      # only use top 20000 unigrams and bigrams
    top_nes = 10     # only use top 1000 named entities

    # TODO model hyperparams
    num_datapoints_for_model = 10000
    n_trees = 10
    model = RandomForestClassifier(n_estimators=n_trees)

    confusion_matrices_pdf = PdfPages(util.home_dir + '/confusion_matrices.pdf')
    seaborn.set_style("darkgrid")
    seaborn.set_context("paper")


    status_ids_order = None

    X, y = concat_features()
    y_pred = train(X, y)
    print(accuracy_score(y_pred))

