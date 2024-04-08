import re
import pandas as pd
import pickle
import nltk
import sklearn
f = open("stop_hinglish.txt", "r")
stopwords = f.read().split("\n")
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def preprocess(data):
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s(?:[ap]m)\s-\s"

    messages = re.split(pattern, data)[1:]
    
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({"user_message":messages, "message_date":dates})

    #converting message_date type
    df["message_date"] = pd.to_datetime(df["message_date"], format = "%d/%m/%Y, %H:%M %p - ")
    df.rename(columns = {"message_date":"date"}, inplace = True)

    # separate users and messages
    users = []
    messages = []

    for message in df["user_message"]:
        entry = re.split("([\w\W]+?):\s", message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append("group_notification")
            messages.append(entry[0])

    df["user"] = users
    df["message"] = messages
    df.drop(columns = ["user_message"], inplace = True)


    df["year"] = df["date"].dt.year

    df["month_num"] = df["date"].dt.month

    df["month"] = df["date"].dt.month_name()

    df["day"] = df["date"].dt.day

    df["day_name"] = df["date"].dt.day_name()

    df["only_date"] = df["date"].dt.date

    df["hour"] = df["date"].dt.hour

    df["minute"] = df["date"].dt.minute

    period = []
    for hour in df[["day_name", "hour"]]["hour"]:
        if hour == 23:
            period.append(str(hour) + "-" + str("00"))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour+1))
        else:
            period.append(str(hour) + "-" + str(hour+1))

    df["period"] = period

    def preprocess_text(text):

        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)
    
    model = pickle.load(open("LogisticRegressionClassifier.pkl", "rb"))
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))

    def predict(text):
        preprocessed = preprocess_text(text)
        vector = tfidf.transform([preprocessed]).toarray()
        output = model.predict(vector)[0]
        return output


    df["sentiment"] = df["message"].apply(predict)

    return df