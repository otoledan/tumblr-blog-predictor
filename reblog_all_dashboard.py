from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import unittest
import mock
import json
import pytumblr
from urllib.parse import parse_qs
import sys
import csv
from multiprocessing import Process, Queue
import re
import urllib
import random

import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
from keras.models import load_model
from pathlib import Path
import sys


def downloader(folder, image_url):
    file_name = image_url.split("/")[-1]
    full_file_name = "./" + str("images") + "/" + str(file_name)

    my_file = Path(full_file_name)

    if my_file.is_file():
        print("file exists")
    else:
        try:
            resp = urllib.request.urlopen(image_url)
            respHtml = resp.read()
            binfile = open(full_file_name, "wb")
            binfile.write(respHtml)
            binfile.close()
        except:
            print("exception occurred")

    return file_name

    #urllib.request.urlretrieve(image_url,full_file_name)


def get_dict_on_blog():
    reader = csv.reader(open("./on_blog.csv"))
    x = True
    count = 0
    dict_of_posts = dict();
    for row in reader:
        if x:
            dict_of_posts[row[0]] = row[1]
            x = False
        else:
            x = True
        count = count + 1
    return dict_of_posts

def get_dict_not_on_blog():
    reader = csv.reader(open("./not_on_blog.csv"))
    x = True
    count = 0
    dict_of_posts = dict();
    for row in reader:
        if x:
            dict_of_posts[row[0]] = row[1]
            x = False
        else:
            x = True
        count = count + 1
    return dict_of_posts

def remove_dups(on_blog, not_on_blog):
    print(len(not_on_blog.keys()))
    for x in on_blog:
        if x in not_on_blog:
            del not_on_blog[x]
    print(len(not_on_blog.keys()))
    print(len(on_blog.keys()))

    with open('not_on_blog_fix.csv','w', newline='') as f:
        w = csv.writer(f)
        w.writerows(not_on_blog.items())

    with open('on_blog_fix.csv','w', newline='') as f:
        w = csv.writer(f)
        w.writerows(on_blog.items())

def right_all(client, on_blog, name):
    not_on_blog = dict()
    count = 1
    start_from = False
    offset = 0

    blog = get_followers(client, name)

    while len(not_on_blog) < len(on_blog):
        get_posts(offset, on_blog, not_on_blog, blog)
        offset = offset + 20

        print(str(count) + " : " + str(len(not_on_blog)))
        count = count + 1

    with open('combined_new.csv','w', newline='') as f:
        w = csv.writer(f)
        w.writerows(not_on_blog.values())
        w.writerows(on_blog.values())

def reblog_all(client, on_blog, name):
    #id = enter id number of post you want to start reblogging from
    length = 20
    dict_of_posts = dict()
    count = 1
    start_from = False

    while length == 20:
        x = client.dashboard(reblog_info=True, since_id=id)[u'posts']
        length = len(x);

        for i in range(0, length):
            if x[i][u'summary'] not in dict_of_posts and x[i][u'summary'] not in on_blog:
                hash = x[i][u'summary']
                dict_of_posts[hash] = x[i]

        id = x[1][u'id']
        print(str(count) + " : " + str(len(dict_of_posts)))
        count = count + 1

    #add body to be posted in html
    body=""
    #add tags to add array
    tags=[]
    keys = list(dict_of_posts.keys())

    for i in range(0, len(keys)):
        post = dict_of_posts[keys[i]]
        id = post['id']
        reblog_key = post['reblog_key']
        client.reblog(name, comment=body, tags=tags, id=id, reblog_key=reblog_key);

        print(str(((i+1)/len(keys))*100) + "%")

def get_all_posts(client, name):
    length = 20
    offset = 0
    count = 1
    dict_of_posts = dict()

    while length == 20:
        x = client.posts(name, offset=offset, reblog_info=True)
        if len(x) > 0:
            x = x[u'posts']
            length = len(x);
            print(length)
            for i in range(0, length):
                sum = clean_string(x[i][u'summary'])

                if sum not in dict_of_posts and len(sum) > 0:
                    hash = sum
                    if 'reblogged_from_name' in x[i] and len(x[i]['reblogged_from_name']) > 0:
                        reblog = x[i]['reblogged_from_name']
                        dict_of_posts[hash] = [hash, reblog ,"reblog"]

                        if 'photos' in x[i]:
                            file_name = downloader('reblog',x[i]['photos'][0]['original_size']['url'])
                            dict_of_posts[hash] = [hash, reblog, file_name,"reblog"]
                        else:
                            dict_of_posts[hash] = [hash, reblog, "", "reblog"]
                    else:
                        if 'photos' in x[i]:
                            file_name = downloader('reblog',x[i]['photos'][0]['original_size']['url'])
                            dict_of_posts[hash] = [hash, reblog, file_name,"reblog"]
                        else:
                            dict_of_posts[hash] = [hash, name, "", "reblog"]


            print(str(count) + " : " + str(len(dict_of_posts)))
            count = count + 1
            offset = offset + 20

    return dict_of_posts

def clean_string(word):
    word = word.replace('“','"').replace('”','"')
    word = word.replace('‘',"'").replace('’',"'")
    word = re.sub(r"[^\w\d'\s]+",'',word)
    word = word.lower()

    word = ''.join([x for x in word if ord(x) < 128])
    return word

def get_followers(client, blog_names):
    blog = []

    length = 20
    offset = 0

    while length == 20:
        x = client.blog_following(blog_names, offset=offset)['blogs']

        if len(x) > 0:
            length = len(x)

            for y in x:
                blog.append(y['name'])

            offset = offset + 20

    return blog

def get_posts(offset, on_blog, not_on_blog, blogs):
    for blog in blogs:
        x = client.posts(blog, offset=offset)
        if len(x) != 0:
            x = x[u'posts']

            for y in x:
                sum = clean_string(y[u'summary'])
                if sum not in on_blog and sum not in not_on_blog:
                    hash = sum
                    if 'photos' in y:
                        file_name = downloader('reblog',y['photos'][0]['original_size']['url'])
                        dict_of_posts[hash] = [hash, blog, file_name, "not on blog"]
                    else:
                        dict_of_posts[hash] = [hash, blog, "", "not on blog"]

def most_notes(client, name):
    length = 20
    offset = 0
    count = 1
    dict_of_posts = dict()
    f = open("most_notes.txt", "w")

    while length == 20:
        x = client.posts(name, offset=offset)[u'posts']
        length = len(x);

        for i in range(0, length):
            if x[i][u'summary'] not in dict_of_posts:
                hash = x[i][u'summary']
                dict_of_posts[hash] = x[i]
                date =  x[i][u'date'].split()[1]
                notes = x[i][u'note_count']
                f.write(date + '\t' + str(notes) + "\n")


        print(str(count) + " : " + str(len(dict_of_posts)))
        count = count + 1
        offset = offset + 20

    return dict_of_posts


def reblog_prediction(client, id_num):
    id = int(id_num) #enter id number to start from on dashboard
    x = client.dashboard(reblog_info=True, since_id=id)[u'posts']
    #load our saved model
    model = load_model('my_model.h5')

    # load tokenizer
    tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    encoder = LabelBinarizer()
    encoder.classes_ = ["reblog", "not on blog"]

    labels = np.array(["reblog", "not on blog"])

    x_data = []

    length = len(x)

    for i in range(0, length):
        str = clean_string(x[i]['blog_name'] + " " + x[i][u"summary"])
        x_data.append(str)

    x_data_series = pd.Series(x_data)
    x_tokenized = tokenizer.texts_to_matrix(x_data_series, mode='tfidf')

    i=0
    for x_t in x_tokenized:
        prediction = model.predict(np.array([x_t]))
        predicted_label = labels[np.argmax(prediction[0])]
        print("Summary: " + x_data[i] +  "\nPredicted label: " + predicted_label)
        i += 1

if __name__ == "__main__":
    #paste the API tokens from tumblr OAuth
    client = pytumblr.TumblrRestClient(
         #Add tumblr API token codes
    )

    if (len(sys.argv) == 3):
        if (sys.argv[1] == "get_data"):
            name = sys.argv[2]

            on_blog = get_all_posts(client, name)
            right_all(client, on_blog, name)

        if (sys.argv[1] == "prediction"):
            id = sys.argv[2]
            reblog_prediction(client, id)

        else:
            print("Please enter either get_blogs or prediction as the 1st argument")
            print("Please enter a blog name or id respectively")

            #on_blog = get_dict()
            #most_notes(client, name)

            #reblog_all(client, on_blog, name)

            #on_blog = get_dict_on_blog()
            #not_on_blog = get_dict_not_on_blog()
            #remove_dups(on_blog, not_on_blog)
