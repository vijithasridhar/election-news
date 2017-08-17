# to open data files
home_dir = '/Users/vijithasridhar/Downloads/Election:News Code'
newsgroups = ['CNN', 'FoxNews', 'NPR', 'NBCNews', 'Breitbart', 'WSJ']
header_names = ["status_id", "status_message", "link_name", "status_type", "status_link", \
           "status_published"]
primary_key = 'status_id'


datafile = home_dir + '/data/%s_facebook_statuses.csv'
lexical_features_file = home_dir + '/features/%s_lexical_features.csv'


# Facebook scraping stuff; hide the app_id and app_secret in private file
app_id = ''
app_secret = ''
try:
    from .private import *
except Exception:
    pass
access_token = app_id + "|" + app_secret


def merge_csvs(original_file, second_file):
    for ng in newsgroups:
        original_data = pd.read_csv(original_file % ng, sep=',')
        more_data = pd.read_csv(second_file % ng, sep=',')

        results = pd.merge(original_data, more_data, on=util.primary_key)
        results.to_csv(original_file, index=False)

