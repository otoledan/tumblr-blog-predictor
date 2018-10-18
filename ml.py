import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
from keras.constraints import maxnorm
from keras.optimizers import SGD

import sys
import csv

def clean(sentence):
    list_of_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    x = sentence.split(" ")

    z = ""
    for y in x:
        if y not in list_of_words:
            z = z + " " + y

    return z


def get_dict_on_blog():

    #if model already exists uncomment the below line

    #return pd.read_pickle("file_name")
    reader = csv.reader(open("./combined_new.csv"))
    dict_of_posts = dict()

    for row in reader:
        dict_of_posts[str(row[1]) + ' ' + str(row[0])] = str(row[2])

    df = pd.DataFrame(columns=['posts', 'category'])
    l = list(dict_of_posts.keys())
    length = len(l)

    for i in range(0, length):
        #if len(l[i].split(" ")) >= 5:
            df.loc[i] = [l[i], dict_of_posts[l[i]]]
            print(i/length,)

    df = df.sample(frac=1).reset_index(drop=True)

    print(df)
    df.to_pickle("file_name")

    return df

def create_model(vocab_size):
	# create model
	model = Sequential()
	model.add(Dense(64, input_shape=(vocab_size,), kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(32, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.8))
	model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

if __name__ == "__main__":
    # For reproducibility
    np.random.seed(1237)

    # We have training data available as dictionary filename, category, data
    data = get_dict_on_blog()

    #data = pd.DataFrame({"posts": d.keys(), "category": d.items()})
    #print(data)
    train_size = int(len(data) * .8)

    train_posts = data['posts'][:train_size]
    train_tags = data['category'][:train_size]

    test_posts = data['posts'][train_size:]
    test_tags = data['category'][train_size:]

    num_labels = 2
    vocab_size = 5000
    batch_size = 100

    # define Tokenizer with Vocab Size
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_posts)

    x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)



    model = create_model(vocab_size)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=3,
        verbose=1,
        validation_split=0.1)

    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1])

    text_labels = encoder.classes_

    for i in range(10):
        prediction = model.predict(np.array([x_test[i]]))
        predicted_label = text_labels[np.argmax(prediction[0])]
        print('Actual label:' + test_tags.iloc[i])
        print("Predicted label: " + predicted_label)

    # creates a HDF5 file 'my_model.h5'
    model.model.save('my_model.h5')

    # Save Tokenizer i.e. Vocabulary
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
