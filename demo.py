
import pandas as pd
#import preprocessor as p
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
#from pandas.tools.plotting import table




from emotion_predictor import EmotionPredictor

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns


model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)

lst = []

'''
with open('data.csv', encoding="utf8") as f:
    reader = csv.reader(f,delimiter=",")
    for row in reader:
        array = row[1].split(',')
        tweet = array[0]
        cleaned_tweets = re.sub(r'[?|$|.|!]',r'',tweet)
        some = re.sub(r'[^a-z A-Z]',r'',cleaned_tweets)
        some = some[3:]
        lst.append(some.lower())
'''
'''
for i in lst :
    print(i)
'''
conversation = " Hotel Management Bot Transcript     User says : Hi message from bot : Can you please provide your Email ID ?   User says : hrishikeshhere@gmail.com message from bot : Hello, would you like to check one of these options ? Book a table Order Chinese Order Dinner Register/Update profile  User says : Update Profile message from bot : Please let me know your preferred food choices User says: Italian Pasta is my favourite "

tweet = conversation
cleaned_tweets = re.sub(r'[?|$|.|!]',r'',tweet)
some = re.sub(r'[^a-z A-Z]',r'',cleaned_tweets)
some = some[3:]
lst.append(some.lower())
print("lst ",lst)
##    

tweets = lst
predictions = model.predict_classes(tweets)
print(predictions, '\n',)


df=pd.DataFrame(predictions,columns=['emotion'])
#print(df.describe)


count_Joy = len(df[predictions['Emotion'] == 'Joy'])
count_Surprise = len(df[predictions['Emotion'] == 'Surprise'])
count_Anger = len(df[predictions['Emotion'] == 'Anger'])
count_Sad = len(df[predictions['Emotion'] == 'Sad'])
count_Fear = len(df[predictions['Emotion'] == 'Fear'])

slices = [count_Joy,count_Surprise,count_Anger,count_Sad,count_Fear]
activities = ['Joy','Surprise','Anger','Sad','Fear']
cols = ['c','m','r','b','y']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        autopct='%1.1f%%')

plt.title('Emotion Pie Chart - Tweets analysis')
plt.show()

''''
probabilities = model.predict_probabilities(tweets)
print(probabilities, '\n')

embeddings = model.embed(tweets)
print(embeddings, '\n')
'''


