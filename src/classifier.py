from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from scipy import sparse
from scipy.io import mmread, mmwrite
import itertools
import util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
import numpy as np
import pandas as pd
from phrasemachine import phrasemachine
from collections import Counter

# for some reason random_state wasn't working properly when I called sample(),
# even with the same size dataframes,
# so I chose to use the same index as in get_vectorizations to sample instead
def sample_by_index(df, index, i, num_one_newsgroup):
    index = list(index)
    indices_for_this_ng = index[i * num_one_newsgroup : (i+1) * num_one_newsgroup]
    return df.ix[indices_for_this_ng]


def concat_features():
    vectorized_features = get_vectorizations_for_all_classes()
    print("Combining all features into data and labels")
    all_lexical_feats = None
    labels = []

    for i, ng in enumerate(util.newsgroups):
        lexical_feats = pd.read_csv(util.lexical_features_file % ng, sep=',', encoding='utf8') 
        print('File %s shape %s' % (i, lexical_feats.shape))
        lexical_feats = sample_by_index(lexical_feats, vectorized_features.index, i, \
                (num_datapoints_for_model // len(util.newsgroups))) 
        labels.append(lexical_feats.shape[0])

        if all_lexical_feats is not None:
            all_lexical_feats = pd.concat([all_lexical_feats, lexical_feats], axis=0)
        else:
            all_lexical_feats = lexical_feats
    
    assert all_lexical_feats.index.equals(vectorized_features.index), "Error: sampling different datapoints " \
            + "from the feature matrices!" 

    # delete the first column because they're all the status IDs, but save that
    status_ids_order = all_lexical_feats[util.primary_key]    
    all_lexical_feats.drop(util.primary_key, axis=1, inplace=True)
    vectorized_features = vectorized_features.fillna(0)

    X = pd.concat((all_lexical_feats, vectorized_features), axis=1)
    # kept track of just how many datapoints for each newsgroup there was
    y = np.repeat(range(1, len(util.newsgroups) + 1), labels)

    feature_names = X.columns
    np.savetxt('feature_names.txt', X.columns, fmt='%5s', delimiter=',')    
    mmwrite('X.csv', sparse.csc_matrix(X.values))
    np.savetxt('y.csv', y, delimiter=',')
    status_ids_order.to_csv('status_ids_order.csv')
    return X, y


def vectorize(vocabulary, list_of_strings):
    mat = None
    for i, string in enumerate(list_of_strings):
        vector = []
        for phrase in vocabulary:
            vector.append(1 if phrase in string else 0)
        vector = sparse.csr_matrix(vector)
        if mat is not None:
            mat = sparse.vstack((mat, vector))
        else:
            mat = vector

    return mat


# returns the named entities and ngrams (vectorized) into a dataframe
def get_vectorizations_for_all_classes():

    # concatenate all headlines into a 1D array so you can extract all ngrams/NEs from them
    all_headlines = None

    for i, ng in enumerate(util.newsgroups):

        # sample datapoints from all the headlines, since you have way too many headlines to use in a model
        # check for NA headlines because you dropped a few links that didn't have headline names when computing features
        # lowercase to do NER without worrying about case (CountVectorizer doesn't work otws)
        all_link_data = pd.read_csv(util.datafile % ng, sep=',', encoding='UTF-8')
        all_link_data = all_link_data[all_link_data['link_name'].notnull()]
        print('File %s shape %s' % (i, all_link_data.shape))
        all_link_data = all_link_data.sample(n=(num_datapoints_for_model // len(util.newsgroups)), \
                random_state=i)['link_name'].str.lower()
        if all_headlines is not None:
            all_headlines = pd.concat([all_headlines, all_link_data], axis=0)
        else:
            all_headlines = all_link_data

    # get named entities with phrasemachine
    print("Generating named entities for all data")
    phrases = sum((phrasemachine.get_phrases(h)['counts'] for h in all_headlines), \
                        Counter()).most_common(top_nes)
    phrases = [str(k[0]) for k in phrases]
    ner = vectorize(phrases, all_headlines)
    # needs pandas 0.20.0+
    ner = pd.SparseDataFrame(data=ner, columns=phrases, index=all_headlines.index)

    # ngrams
    print("Generating ngrams for all data")
    vectorizer = CountVectorizer(ngram_range=(min_n, max_n), analyzer=analyzer, max_features=top_ngrams)
    ngrams = vectorizer.fit_transform(all_headlines)
    ngrams = pd.SparseDataFrame(data=ngrams, columns=vectorizer.get_feature_names(), \
            index=all_headlines.index)

    return pd.concat([ner, ngrams], axis=1)
    


def train(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(   \
                    X, y, test_size=0.2, random_state=0)
 
    # clf = RandomForestClassifier()
    # params_grid = {'n_estimators': [40, 60, 80, 100, 120]}
    # print("Params searched are %s" % params_grid)
    # grid_clf = GridSearchCV(clf, params_grid, cv=10)
    # grid_clf.fit(X_train, y_train)
    model = RandomForestClassifier(n_estimators=120) #grid_clf.best_estimator_
    # print("Best params from grid search are %s" % grid_clf.best_params_)
    # print("Best score from grid search on left-out data was %s" % grid_clf.best_score_)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.get_params())

    print("Training error: %.6f" % accuracy_score(model.predict(X_train), y_train, 'Training data confusion matrix'))
    print("Test error: %.6f" % accuracy_score(y_pred, y_test, 'Test data confusion matrix'))
    
    print_best_features(model.feature_importances_, feature_names)


def print_best_features(coef, feature_names):
    #zip two lists together to iterate through them simultaneously
    #sort the two lists by the values in the first (the coefficients)
    zipped = zip(coef, feature_names)				
    zipped.sort(key = lambda t: t[0], reverse=True)

    print("\nMOST IMPORTANT FEATURES:")
    for (weight, word) in zipped[:40]:
        print("{}\t{:.6f}".format(word, weight))

    print("\nLEAST IMPORTANT FEATURES:")
    for (weight, word) in zipped[:-40:-1]:
        print("{}\t{:.6f}".format(word, weight))


def accuracy_score(y_pred, y_true, title):
    # plot confusion matrix for classes
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(title, cnf_matrix)
    np.savetxt(util.home_dir + '/' + title + '.csv', cnf_matrix, delimiter=',')
    np.set_printoptions(precision=2)
    plt.figure()
    if np.unique(y_true).size > 2:
        plot_confusion_matrix(cnf_matrix, classes=util.newsgroups, title=title)
    else:
        plot_confusion_matrix(cnf_matrix, classes=['Liberal', 'Conservative'], title=title)
    confusion_matrices_pdf.savefig(plt.gcf())
    
    return sklearn_accuracy_score(y_true, y_pred)


# taken from sklearn documentation
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.colorbar()        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_sparse(filename, delimiter=','):
    txt = open(filename, 'r')
    l1 = []
    for line in txt:
        l1.append([sparse.coo_matrix(np.loadtxt([line], delimiter=delimiter))])
    txt.close()
    return sparse.bmat(l1)



if __name__ == "__main__":
    # TODO hyperparameters for ngrams/NER
    min_n = 1
    max_n = 2
    analyzer = 'word'
    top_ngrams = 20000      # only use top 20000 unigrams and bigrams
    top_nes = 10000     # only use top 1000 named entities

    # TODO model hyperparams
    num_datapoints_for_model = 10000
    party_prediction = True    # predict e.g. liberal vs conservative instead of Dem vs Rep

    confusion_matrices_pdf = PdfPages(util.home_dir + '/confusion_matrices.pdf')
    seaborn.set_style("darkgrid")
    seaborn.set_context("paper")
    
    status_ids_order = None
    feature_names = None

    # X, y = concat_features()
    # feature_names = X.columns
    X = mmread('X.csv.mtx')
    y = np.genfromtxt('y.csv', delimiter=',')
    with open('feature_names.txt', 'r') as f:
        feature_names = f.read().splitlines()
    print("Successfully loaded feature and label files")

    train(X, y)
    if party_prediction:
        y = np.where(((y == 1) | (y == 3) | (y == 4)), 0, 1)  #liberal is 0
        print("\n\n\nRESULTS FOR PARTY PREDICTION")
        train(X, y)

    confusion_matrices_pdf.close()
    


