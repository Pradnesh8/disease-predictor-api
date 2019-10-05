# Import libraries
import pandas as pd
import sklearn
import numpy as np
from flask import Flask, request, jsonify
import pickle
import nltk
import re
from nltk.corpus import stopwords
from flask_cors import CORS,cross_origin
app=Flask(__name__)
CORS(app)
# Load the model
model = pickle.load(open('diseasedata.pkl', 'rb'))

@app.route('/')
def index():
    return "Hello world"

@app.route('/api', methods=['GET'])
def predict():

    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    #print(stop)
    sno = nltk.stem.SnowballStemmer('english')

    def cleanhtml(sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', sentence)
        return cleantext

    def cleanpunc(sentence):
        '''This function cleans all the punctuation or special characters from a given sentence'''
        cleaned = re.sub(r'[?|@|!|^|%|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
        return cleaned

    def preprocessing(series):
        i = 0
        str1 = " "
        final_string = []
        list_of_sent = []

        for sent in series.values:
            filtered_sent = []
            list_of_sent = []
            sent = cleanhtml(sent)
            sent = cleanpunc(sent)
            for cleaned_words in sent.split():
                if ((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):
                    if (cleaned_words.lower() not in stop):
                        s = (sno.stem(cleaned_words.lower()))
                        filtered_sent.append(s)
            list_of_sent.append(filtered_sent)
            str1 = " ".join(filtered_sent)
            final_string.append(str1)
            i += 1
        return final_string, list_of_sent

    # Get the data from the POST request.
    data = request.args['symptoms']
    data=data.split(',')
    final_string, list_of_symptoms = preprocessing(pd.Series(data))

    all_symptoms = pickle.load(open('symptoms.pkl', 'rb'))

    symptoms_list = []
    for x in final_string:
        symptoms_list.append(list(all_symptoms.keys()).index(x))

    # Make prediction using model loaded from disk as per the data.
    sample_x = [i / i if i in symptoms_list else i * 0 for i in range(len(all_symptoms))]

    sample_x = np.array(sample_x).reshape(1, len(sample_x))

    prediction = model.predict(sample_x)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

