import pickle
import string
import tflearn
import json
import os
import numpy as np
import preprocessor as p
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
from keras.models import model_from_json,Sequential
from keras.layers import Embedding,Dropout,LSTM,Bidirectional,Dense
os.environ['KERAS_BACKEND']='tensorflow'

def evaluate_model(model, testX, testY):
    temp = model.predict(testX)
    y_pred  = np.argmax(temp, 1)
    y_true = np.argmax(testY, 1)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    print("Precision: " + str(precision) + "\n")
    print("Recall: " + str(recall) + "\n")
    print("f1_score: " + str(f1_score) + "\n")
    print(confusion_matrix(y_true, y_pred))
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    return precision, recall, f1_score

def save_model(data,model_type,model,embed_size):
    weight_file_name = "results/" + str(data) + "_" + str(model_type) + "_" + str(embed_size)+ "weight.h5"
    model_file_name =  "results/" + str(data) + "_" + str(model_type) + "_" + str(embed_size)+"model.json"
    model_json = model.to_json()
    with open(model_file_name, "w+") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_file_name)
    print("Saved model to disk")

def blstm(inp_dim,vocab_size, embed_size, num_classes, learn_rate):   

    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size)))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

def train(data_dict, embed_size,data,dump_embeddings=False):
    global NUM_CLASSES,LEARN_RATE,BATCH_SIZE,EPOCHS
    data, trainX, trainY, testX, testY, vocab_processor = data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"], data_dict["vocab_processor"]
    
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping
    
    print("Running Model: " + model_type )
    model = blstm(trainX.shape[1], vocab_size, embed_size, NUM_CLASSES, LEARN_RATE)


    model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, 
                  verbose=1)
    save_model(data,model_type,model,embed_size)
    
    return  evaluate_model(model, testX, testY)

def get_train_test(data, x_text, labels):
    global NUM_CLASSES
    X_train, X_test, Y_train, Y_test = train_test_split( x_text, labels, random_state=42, test_size=0.10)
    
    post_length = np.array([len(x.split(" ")) for x in x_text])
    #take 95 percentile of size as input dim of text
    max_document_length = int(np.percentile(post_length, 95))

    print("Document length : " + str(max_document_length))
    
    #converting words into integers ie representing text input into vectors
    #max_document is input dim
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    #finds the unique words in corpus
    vocab_processor = vocab_processor.fit(x_text) 
    #convert them into aactual vectors
    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))
    
    trainY = np.asarray(Y_train)
    testY = np.asarray(Y_test)
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)
    
    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    
    return data_dict

def load_data(filename):
    global HASH_REMOVE
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    for i in range(len(data)):
        x_text.append(list(map(str.lower,data[i]['text'])))
        labels.append(data[i]['label'])
    return x_text,labels

def get_data(data, oversampling_rate):
    global NUM_CLASSES
    file_name = 'data/wiki_data.pkl'

    x_text, labels = load_data(file_name) 
 
    #increase size of bully data in dataset
    NUM_CLASSES = 2
    bully = [i for i in range(len(labels)) if labels[i]==1]
    x_text = x_text + [x_text[x] for x in bully]*(oversampling_rate-1)
    labels = list(labels) + [1 for i in range(len(bully))]*(oversampling_rate-1)

    print("Counter after oversampling")
    from collections import Counter
    print(Counter(labels))
    
    #remove punctuations
    filter_data = []
    for text in x_text:
        filter_data.append("".join(l for l in text if l not in string.punctuation))
        
    return filter_data, labels

def run_model(data, oversampling_rate, embed_size):    
    x_text, labels = get_data(data, oversampling_rate)
    data_dict = get_train_test(data,  x_text, labels)
    precision, recall, f1_score = train(data_dict, embed_size,data)

def predict():
    file = "results/temp_wiki_blstm_100model.json"
    weight = "results/temp_wiki_blstm_100weight.h5"
    data = "wiki"
    oversampling_rate = 3
    
    #load the model
    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
 
    # load weights into new model
    loaded_model.load_weights(weight)
    print("Loaded model from disk")

    
    x_text, labels = get_data(data, oversampling_rate)
    data_dict = get_train_test(data,  x_text, labels)
    print("Printing model summary ")
    #evaluate_model(loaded_model, data_dict['testX'], data_dict['testY'])
    #print(x_text[6])

    post_length = np.array([len(x.split(" ")) for x in x_text])
 
    max_document_length = int(np.percentile(post_length, 95))

    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)
    inp = 0
    while inp != "":
        inp = input()
        text = ""
        for c in inp:
            if c not in string.punctuation:
                text+=c
        print(text)
        text_list = [text]
        text_vector = np.array(list(vocab_processor.transform(text_list)))

        ans = loaded_model.predict(text_vector)
        print(ans)
        if ans[0][0] > ans[0][1]:
            print("no")
        else:
            print("yes")



#main function
EPOCHS = 10
BATCH_SIZE = 1024
MAX_FEATURES = 2
NUM_CLASSES = None
DROPOUT = 0.25
LEARN_RATE = 0.01
HASH_REMOVE = None
output_folder_name = "results/"
oversampling_rate = 3

data = "wiki"
model_type = "blstm"
embed_size = 100

n = input()
if n == "train":
    run_model(data, oversampling_rate, embed_size)
else:
    predict()