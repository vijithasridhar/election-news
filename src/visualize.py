import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import util
import seaborn
import csv

def visualize():

    #final_pdf = PdfPages(util.home_dir + '/visualizations.pdf')
    final_pdf = PdfPages(util.home_dir + '/visualizations2.pdf')
    
    seaborn.set_style("darkgrid")
    seaborn.set_context("paper")
    
    # get list of features to plot based on the file you're plotting from
    features_list = None
    features_file_to_plot = util.home_dir + '/features/%s_more_lexical_features.csv'
    with open(features_file_to_plot % util.newsgroups[0], newline='') as f:
        csv_reader = csv.reader(f)
        features_list = next(csv_reader)[1:]

    # ['flesch_reading_ease', 'avg_word_length', 'compound_sentiment', \
    #        'positive_sentiment', 'negative_sentiment', 'num_hyperbolic_words']

    for feat in features_list:
        plt.figure()
        
        for ng in util.newsgroups:
            # merge dataframes
            all_link_data = pd.read_csv(util.datafile % ng, sep=',', names=util.header_names)
            lexical_feats = pd.read_csv(features_file_to_plot % ng, sep=',')

            results = pd.merge(all_link_data, lexical_feats, on=util.primary_key)

            x, y = group_data_by_time(results, feat, 'D', k_moving_avg)
            plt.plot(x, y, label=ng)  
        
        plt.title(feat.capitalize().replace('_', ' ') + ' for headlines of various newsgroups')
        plt.xlabel("Date")

        if feat != 'avg_word_length' and feat != 'num_hyperbolic_words':
            plt.ylabel(feat.capitalize().replace('_', ' ') + ' score')
        else:
            plt.ylabel(feat.capitalize().replace('_', ' '))
        plt.legend(loc='upper right')
        final_pdf.savefig(plt.gcf())
        
    final_pdf.close()


# manipulate timestamps / group by month for better visualizing
def group_data_by_time(results, feat, timeframe, k_moving_avg):
    results['status_published'] = results['status_published'].map(pd.to_datetime)
    grouped_results = results.groupby(pd.TimeGrouper(freq=timeframe, key='status_published'))
    x, y = [], []
    for d, group in grouped_results:
        x.append(d)
        y.append(group[feat].rolling(k_moving_avg).mean())
    return x, y


if __name__ == "__main__":
    # TODO HYPERPARAMETER
    k_moving_avg = 14       # take 14 days moving average when displaying trends in the data
    visualize()
    

