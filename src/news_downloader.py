# import some Python dependencies
import os
import requests
import json
import datetime
import csv
import time

def request_until_succeed(url):
    success = False
    while success is False:
        try: 
            response = requests.get(url, timeout=5)            
            if response.status_code == 200:
                success = True
        except Exception as e:
            print(e)
            time.sleep(5)
            
            print("Error for URL %s: %s" % (url, datetime.datetime.now()))

    return response.text


def getFacebookPageFeedData(page_id, access_token, num_statuses):

    # construct the URL string
    base = "https://graph.facebook.com/v2.9"
    node = "/" + page_id + "/feed" # changed
    parameters = "/?fields=message,link,created_time,name,type&access_token=%s" % access_token
    url = base + node + parameters
    
    # retrieve data
    data = json.loads(request_until_succeed(url))
    #print json.dumps(data, indent=4, sort_keys=True)
    
    return data    


def processFacebookPageFeedStatus(status):
    status_id = status['id']
    status_message = '' if 'message' not in status.keys() else status['message'].encode('utf-8')
    link_name = '' if 'name' not in status.keys() else status['name'].encode('utf-8')
    status_type = status['type']
    status_link = '' if 'link' not in status.keys() else status['link']
    
        
    status_published = datetime.datetime.strptime(status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
    
    # return a tuple of all processed data
    return (status_id, status_message, link_name, status_type, status_link,
           status_published)


def is_link(status, page_id):
    if not status[3] == 'link':
        return False
    if page_id.lower() == "cnn" and not "cnn.it" in status[4]:
        return False
    return True



def scrapeFacebookPageFeedStatus(page_id, access_token):
    with open(util.datafile % page_id, 'w', newline='') as file:
        w = csv.writer(file)
        w.writerow(util.header_names)
        
        has_next_page = True
        num_processed = 0   # keep a count on how many we've processed
        scrape_starttime = datetime.datetime.now()
        
        print("Scraping %s Facebook Page: %s\n" % (page_id, scrape_starttime))
        
        statuses = getFacebookPageFeedData(page_id, access_token, 100)
        try:
            while has_next_page:
                for status in statuses['data']:
                    status = processFacebookPageFeedStatus(status)
                    if status[5].date() < datetime.date(2013, 7, 1):
                        has_next_page = False
                        break

                    if is_link(status, page_id):
                        w.writerow(status)
                    
                        # output progress occasionally to make sure code is not stalling
                        num_processed += 1
                        if num_processed % 1000 == 0:
                            print("%s Statuses Processed: %s" % (num_processed, datetime.datetime.now()))
                        
                # if there is no next page, we're done.
                if 'paging' in statuses.keys() and 'next' in statuses['paging'].keys():
                    statuses = json.loads(request_until_succeed(statuses['paging']['next']))
                else:
                    has_next_page = False
        except KeyboardInterrupt:
            print("Processed %s statuses " % num_processed)
                
        
        print("Done!\n%s Statuses Processed in %s" % (num_processed, datetime.datetime.now() \
                - scrape_starttime))

if __name__ == "__main__":
    import util
    for ng in util.newsgroups:
        if not os.path.isfile(util.datafile % ng):
            scrapeFacebookPageFeedStatus(ng, util.access_token)

