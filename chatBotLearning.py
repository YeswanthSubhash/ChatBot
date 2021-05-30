import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#print(intents)

for intent in intents['intents']:
    #print(intent)
    for pattern in intent['patterns']:
        #print(pattern)
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        #print(w)
        words.extend(w)
        #print(w)

        # add documents in the corpus
        documents.append((w, intent['tag']))
        #print(documents)

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
#print(words)
#words = sorted(list(set(words)))
#print(words)
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
# words = all words, vocabulary
#print (len(words), "unique lemmatized words", words)

pickled_obj=pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#print("Pickled Object",pickled_obj)

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print(pattern_words)
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

print(np.array(training))