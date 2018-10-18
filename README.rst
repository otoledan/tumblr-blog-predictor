#Tumblr Blog Predictor

## Description
These python programs use the text descriptions from previous posts on any tumblr blog and predict which posts on their dashboard will be reblogged. 

## First Use

### Prerequisites 
Install Keras (https://keras.io/#installation)
Install Tensorflow (https://www.tensorflow.org/install/)

### Get the Tumblr Authorization Keys

#### Register your blog to get the consumer and secret keys
http://www.tumblr.com/oauth/register

#### Get remaining authorization key from Tumblr
https://api.tumblr.com/console/calls/user/info


### Place keys in file
Using the keys just accessed paste them into the `reblog_all_dashboard.py` file in the following format on the bottom of the file.

```
client = pytumblr.TumblrRestClient(
        '<consumer_key>',
        '<consumer_secret>',
        '<oauth_token>',
        '<oauth_secret>',
    )
```

## Use
Run `reblog_all_dashboard.py` with the following arguments `get_data` and `<blog name>`. 

This will generate two CSV files which will be used to create a predictive model for which posts would be reblogged or not reblogged.

If a model has been generated and wants to be reused uncomment the line `return pd.read_pickle("file_name")` in `ml.py`.

Run `ml.py`. 

Now get the id of the post you want to start evaluating from on your Dashboard. Paste this id number in the `reblog_prediction` function in `reblog_all_dashboard.py`.

Run `reblog_all_dashboard.py` with the following arguments `prediction` and `<blog name>`. 



## Credits
Made using the Tumblr API V2 (https://www.tumblr.com/docs/en/api/v2) and the python client for the Tumblr API.

Text classification code was based on the tutorial on OpenCodez (https://www.opencodez.com/python/text-classification-using-keras.htm) titled Simple Text Classification using Keras Deep Learning Python Library.
