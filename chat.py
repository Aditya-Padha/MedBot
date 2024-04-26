import json
import random
import nltk
import numpy as num
import pickle
from nltk.stem import WordNetLemmatizer
import tensorflow as tensorF
import chatx
import Hospital
import Doctor
import drug


with open('data.json', 'r') as file:
    ourData = json.load(file)

with open('newWords.pkl', 'rb') as file:
    newWords = pickle.load(file)

with open('ourClasses.pkl', 'rb') as file:
    ourClasses = pickle.load(file)


loaded_model = tensorF.keras.models.load_model('model.h5')
lm = WordNetLemmatizer()


def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
      for idx, word in enumerate(vocab):
        if word == w:
          bagOwords[idx] = 1
    return num.array(bagOwords)


def pred_class(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = loaded_model.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
      newList.append(labels[r[0]])
    return newList


def getRes(firstlist, fJson):
    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    for i in listOfIntents:
      if i["tag"] == tag:
        ourResult = random.choice(i["responses"])
        break
    return ourResult



def chatRes(data):
    newMessage = data.lower()
    intents = pred_class(newMessage, newWords, ourClasses)
    symptom_data = json.loads(newMessage)
    symptoms = symptom_data.get("title")
    if "/" in symptoms:
        print(symptoms.split('/')[-1])
        if symptoms.split('/')[-1] == 'doctor':
            newMessage = symptoms.split('/')[0]
            with open("history.txt", "r") as hist:
                special = hist.read()
            if newMessage.split(',')[-1].isdigit() and len(newMessage.split(',')[-1]) == 6:
                print(newMessage.split(',')[0])
                return Doctor.recommend_doctor(special, newMessage.split(',')[1], newMessage.split(',')[0])

        elif symptoms.split('/')[-1] == 'hosp':
            newMessage = symptoms.split('/')[0]
            if newMessage.split(',')[-1].isdigit() and len(newMessage.split(',')[-1]) == 6:
                print(newMessage.split(',')[0])
                return Hospital.recommend_hospital(newMessage.split(',')[0], newMessage.split(',')[1], newMessage.split(',')[2])

        elif symptoms.split('/')[-1] == 'drug':
            newMessage = symptoms.split('/')[0]
            with open("predicted_disease.txt", "r") as dis:
                disease = dis.read()
            return drug.recommend_drug(disease, newMessage.split(',')[0], newMessage.split(',')[1])
    if symptoms.split(',')[-1].isdigit() and len(symptoms.split(',')[-1]) == 6:
        return Hospital.recommend_hospital(symptoms.split(',')[0], symptoms.split(',')[1], symptoms.split(',')[2])
    if intents[0] == "symptom":
        result = chatx.chatRes(symptoms)
        with open("history.txt", "w") as hist:
            hist.write(result.get('Specialist_Required')[0])
    else:
        result = getRes(intents, ourData)
    return result

