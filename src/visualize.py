import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import util
import seaborn
import csv

def visualize():

    final_pdf = PdfPages(util.home_dir + '/visualizations.pdf')
    
    seaborn.set_style("darkgrid")
    seaborn.set_context("paper")
    
    # get list of features to plot based on the file you're plotting from
    features_list = None
    features_file_to_plot = util.lexical_features_file
    #features_file_to_plot = util.home_dir + '/features/%s_more_lexical_features.csv'

    with open(features_file_to_plot % util.newsgroups[0], newline='') as f:
        csv_reader = csv.reader(f)
        features_list = next(csv_reader)[1:]


    for feat in features_list:
        plt.figure()
        
        for ng in util.newsgroups:
            # merge dataframes to use the publish time as an index
            all_link_data = pd.read_csv(util.datafile % ng, sep=',', names=util.header_names)
            lexical_feats = pd.read_csv(features_file_to_plot % ng, sep=',')
            results = pd.merge(all_link_data, lexical_feats, on=util.primary_key)

            results['status_published'] = results['status_published'].map(pd.to_datetime)
            results = results.set_index('status_published')
            rolling_means = results.resample("1D").mean()
            rolling_means = rolling_means.rolling(window=k_moving_avg, min_periods=1).mean()

            plt.plot(rolling_means.index.values, rolling_means[feat], label=ng)  
        
        plt.title(pretty_name_for_feat(feat) + ' for headlines of various newsgroups')
        plt.xlabel("Date")
        plt.ylabel(pretty_name_for_feat(feat))
        plt.legend(loc='upper right')
        final_pdf.savefig(plt.gcf())
        
    final_pdf.close()



def pretty_name_for_feat(feat):
    feat_pretty_name = feat.split('_')
    if feat[0] == 'count':
        feat_pretty_name.append(feat_pretty_name.pop(0))
    if feat[-1] == 'sentiment' or feat[0] == 'lexical' or feat[0] == 'flesch':
        feat_pretty_name.append('score')
    if feat[-1] == 'words':
        feat_pretty_name.append('count')

    return ' '.join(feat).capitalize()





# manipulate timestamps / do a moving average by k days for better visualizing
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
    

