import pickle
import tensorflow
import pandas as pd
import json
import random
import tflearn
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()

data = pd.read_json(
    "https://raw.githubusercontent.com/Siddesh2801/AI-Chatbot/main/Intents.json?token=GHSAT0AAAAAABSKW6TLEWTS3KOCFBUKBWESYRSFSPA")

try:
    with open("FAQindents.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # divides the complete string multiple 1 word strings
            tokenized_words = nltk.word_tokenize(pattern)
            words.extend(tokenized_words)
            # all FAQ phrases correesponding to a tag are put into a list
            docs_x.append(tokenized_words)
            # docs_y corresponds to which FAQ belongs to which category....
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Stem all the words to their root meaning
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    """ Creating a list with 0s, each 0 represents a label.
      The 0 will be incremented, if we identify some pattern
      as a specific label. (One Hot Encoding)
    """
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        doc_x_stem = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in doc_x_stem:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("FAQindents.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("chatbot.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=595, batch_size=8, show_metric=True)
    model.save("chatbot.tflearn")


def bag_of_words(s, words):
    """
    This function tokenizes and stems 
    the words in the input sentence entered
    by the user and converts it into a bag of
    words.
    """

    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    """
    This function provides a interface
    between the model and the user for 
    easy communication. It is also 
    responsible for selecting the final
    response from the choices provided 
    by the json file 
    """
    print("Start talking with the bot <type quit to exit>")
    while True:
        inp = input("\n You:")
        if(inp.lower() == "quit"):
            break
        results = model.predict([bag_of_words(inp, words)])
        result_index = np.argmax(results)
        pred_label = labels[result_index]

        if(results[0, result_index] > 0.65):
            for tg in data["intents"]:
                if tg['tag'] == pred_label:
                    response = tg['responses']
            print("\n")
            print("Bot: ", random.choice(response))
        else:
            print("\n")
            print("Bot: I don't quite understand, try rephrasing your question")


chat()
