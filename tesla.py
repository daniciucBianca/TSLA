import random
import os
import re
from random import randint
import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.stats
from textblob import TextBlob
import math
# Further Analysis
from nltk.corpus import stopwords

from os import path
from PIL import Image

from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import yfinance as yf
from statistics import mean 
from statistics import fmean 
import statistics

#%matplotlib inline

# DATA :
tesla_data = pd.read_csv(
    r"C:/Users/bianc/OneDrive/Documents/CTI/NB/books - pdfs/js/TSLA.csv")

acv = pd.read_csv(
    r"C:/Users/bianc/OneDrive/Documents/CTI/NB/books - pdfs/js/NFLX.csv")

gm_data = pd.read_csv(
    r"C:/Users/bianc/OneDrive/Documents/CTI/NB/books - pdfs/js/RACE.csv")


# how close the closing price and adj_close are
sns.relplot(x='close', y=('adj_close'), data=tesla_data)

## BETA: check for volatility using 
symbols = ['TSLA', 'RACE']
data = yf.download(symbols, '2022-12-06')['Adj Close']
price_change = data.pct_change()
df_dd = price_change.drop(price_change.index[0])
x = np.array(df_dd['TSLA']).reshape((-1,1))
y = np.array(df_dd['RACE'])
model = LinearRegression().fit(x, y)
print('Beta = ', model.coef_)



df_tesla = tesla_data
df_tesla['close'].plot(figsize=(10, 7))
plt.title("Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel(tesla_data['date'], fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

adjPriceDClosePrice = (
    (tesla_data['adj_close'] - tesla_data['close']))
print(adjPriceDClosePrice)

diffBAdjandClose1 = (
    (acv['adj_close'] - acv['close']))
print(diffBAdjandClose1)



print("first 5 rows: ")
print( tesla_data.tail())
print("Working with the following types of data:")
print(tesla_data.describe())
print(tesla_data.dtypes)

print("Volume:")
tesla_data['volume'].plot()
plt.show()

# http://localhost:8888/notebooks/Downloads/IMBA/4.%20Simple_linear_regression/regression-modeling-simple-linear-regression.ipynb --> idea
df_tesla['meanOfClose'] = df_tesla['adj_close'].mean()
## Calculate SSE
sse = np.sum(np.square(df_tesla['adj_close'] - df_tesla['meanOfClose']))
print(sse)

mse = np.mean(np.square(df_tesla['adj_close'] - df_tesla['meanOfClose']))
print(mse)

rmse = (np.mean(np.square(df_tesla['adj_close'] - df_tesla['meanOfClose']))) ** 0.5
print(rmse)
#minimizing the rmse
y_bar = df_tesla['close'].mean()
x_bar = df_tesla['low'].mean()
std_y = np.std(df_tesla['close'], ddof = 1)
std_x = np.std(df_tesla['low'], ddof = 1)
r_xy = df_tesla.corr().loc['close','low']
beta_1 = r_xy*(std_y/std_x)
beta_0 = y_bar - beta_1*x_bar

df_tesla['Linear_meanOfClose'] = beta_0 + beta_1 * df_tesla['low']
Srmse = np.square(df_tesla['close'] - df_tesla['Linear_meanOfClose']).mean()
print(Srmse)

## not good:
np.random.seed(42)
x = np.linspace(-5, 50, 100)
y = 50 + 2 * x  + np.random.normal(0, 20, size=len(x))
# create a figure
fig = plt.figure(figsize=(15,7))
# get the axis of that figure
ax = plt.gca()
# plot a scatter plot on it with our data
ax.scatter(x, y, c='k')
ax.plot(df_tesla['open'], df_tesla['Linear_meanOfClose'], color='r');


df_tesla_mean = df_tesla['open'].mean()
print(df_tesla_mean)

df_tesla_maxO = df_tesla['open'].max()
print("max of open: ", df_tesla_maxO)
df_tesla_minO = df_tesla['open'].min()
print("min of open: ", df_tesla_minO)


df_tesla_maxC = df_tesla['close'].max()
print("max of close: ", df_tesla_maxC)    
df_tesla_minC = df_tesla['close'].min()
print("min of close: ", df_tesla_minC)


df_tesla_maxH = df_tesla['high'].max()
print("max of high: ", df_tesla_maxH)
df_tesla_minH = df_tesla['high'].min()
print("min of high: ", df_tesla_minH)


df_tesla_maxL = df_tesla['low'].max()
print("max of low: ", df_tesla_maxL)
df_tesla_minL = df_tesla['low'].min()
print("min of low: ", df_tesla_minL)

header =['Max of open', 'Min of open', 'Max of close', 'Min of close', 'Max of high', 'Min of high', 'Max of low', 'Min of low']
data = [df_tesla_maxO, df_tesla_minO,  df_tesla_maxC,df_tesla_minC, df_tesla_maxH, df_tesla_minH , df_tesla_maxL, df_tesla_minL ]
print(data)




# return of stock
stock_return = (
    (tesla_data['open'] - tesla_data['close']) / tesla_data['open']) * 100
print(stock_return)
print("Stock return:")
fig = plt.figure()
stock_return.plot()
plt.show()

# return of stock per date
sns.relplot(x='date', y=stock_return, data=tesla_data)

sns.relplot(x='date', y='adj_close', data=tesla_data)

ten_days = tesla_data.head(10)
print("adjusted closing price for 10 days")
sns.relplot(x= ten_days['adj_close']  , y=ten_days['date'], data=tesla_data, hue=tesla_data['volume'])


ten_dayst = tesla_data.tail(10)
print("closing price for 10 days")
sns.relplot(x=ten_dayst['adj_close'], y= ten_dayst['date'], data=tesla_data, hue=tesla_data['volume'])

# study the Volume ( and return of stocks )
# return of stocks per volume
sns.relplot(x= stock_return, y='volume' , data=tesla_data)
figs = plt.figure()





# volatility
tesla_data['daily_returns'] = (tesla_data['close'].pct_change())
daily_volatility = (tesla_data['daily_returns'].std()) 
print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))

monthly_volatility = (math.sqrt(21) * daily_volatility)
print('Monthly volatility:', '{:.2f}%'.format(monthly_volatility))

annual_volatility = (math.sqrt(252) * daily_volatility) 
print('Annual volatility:', '{:.2f}%'.format(annual_volatility))

# plotting the volatility
plot_data_v = pd.DataFrame({
    "daily": daily_volatility,
    "monthly": monthly_volatility,
    "annualy": annual_volatility
}, index=[""]
)
plot_data_v.plot(kind="bar")
plt.title("Volatily of Tesla")
plt.xlabel("")
plt.ylabel("Total")








#Our goal will be to predict the future price. Using a SLR model.

sns.lmplot(x='low', y='open', data=tesla_data)
plt.show()
sns.lmplot(x='volume', y='open', data=tesla_data)
plt.show()
#sns.lmplot(x='date', y=stock_return , data=tesla_data)
#plt.show()

#intercept and slope coefficients that minimize SSE.
import scipy.stats
# Get the optimal slope and y intercept. # predict the next close price​
def lin_reg(x,y):
    # Using other libraries for standard deviation and the Pearson correlation coefficient.
    # Note that in SLR, the correlation coefficient multiplied by the standard
    # deviation of y divided by standard deviation of x is the optimal slope.
    beta1 = (scipy.stats.pearsonr(x,y)[0])*(np.std(y)/np.std(x))
    
    # The Pearson correlation coefficient returns a tuple, so it needs to be sliced/indexed.
    # The optimal beta is found by: mean(y) - b1 * mean(x).
    beta0 = np.mean(y)-(beta1*np.mean(x)) 
    
    return beta1, beta0
x = tesla_data['close'].values
y = tesla_data['open'].values

pred = lin_reg(x,y) + (lin_reg(x,y)*x)


tesla_data['Preds'] = pred


# Plot showing our linear forecast.  
fig = plt.figure(figsize=(20,20))
# Change the font size of minor ticks label.
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)
# Get the axis of that figure.
ax = plt.gca()
# Plot a scatterplot on the axis using our data.
ax.scatter(x= tesla_data['open'], y=tesla_data['close'], c='k')
ax.plot(tesla_data['open'], tesla_data['close'], color='r'); 

# train the model:
    
tesla_data['open-close']  = tesla_data['open'] - tesla_data['close']
tesla_data['low-high']  = tesla_data['low'] - tesla_data['high']
tesla_data['target'] = np.where(tesla_data['close'].shift(-1) > tesla_data['close'], 1, 0)

plt.pie(tesla_data['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

# normalization:

features = tesla_data[['open-close', 'low-high']]
target = tesla_data['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
  models[i].fit(X_train, Y_train)
 
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print()
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()

#prediction 

tesla_data['new_open'] = (tesla_data['open'] > 350).astype(int)
tesla_data['new_open'].value_counts() 

x = tesla_data[['open','close','adj_close','new_open']].values
y = tesla_data['volume'].values

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

tesla_data['prediction'] = y_pred
sns.lmplot(x='volume', y='prediction', data=tesla_data, hue='new_open')

#residuals

tesla_data['Residuals'] = abs(tesla_data['volume']-tesla_data['prediction'])
tesla_data['Residuals'].mean()
tesla_data['Residuals'] = tesla_data['volume'] - tesla_data['prediction']
sns.distplot(tesla_data['Residuals'])
sns.lmplot(x='open', y='Residuals', data=tesla_data)


# TEXT :
f = open(r"C:\Users\bianc\OneDrive\Documents\CTI\NB\books - pdfs\js\tesla_text.txt")
text = f.read()

# Regex cleaning
expression = "[^a-zA-Z0-9 ]"  # keep only letters, numbers and whitespace
cleantextCAP = re.sub(expression, '', text)  # apply regex
cleantext = cleantextCAP.lower()  # lower case

# Save dictionaries for wordcloud
tesla_file = open("Output.txt", "w")
tesla_file.write(str(cleantext))
tesla_file.close()

# Count and create dictionary
dat = list(cleantext.split())
dict1 = {}
for i in range(len(dat)):
    print(i)
    word = dat[i]
    dict1[word] = dat.count(word)

# Unsorted speech constituents in dictionary as dict1
keys = list(dict1)
filtered_words = [
    word for word in keys if word not in stopwords.words('english')]
dict2 = dict((k, dict1[k]) for k in filtered_words if k in filtered_words)

print(filtered_words)

# Resort in list
# Reconvert to dictionary
# length is length of highest consecutive value vector


def SequenceSelection(dictionary, length, startindex=0):

    # Test input
    lengthDict = len(dictionary)
    if length > lengthDict:
        return print("length is longer than dictionary length")
    else:
        d = dictionary
        items = [(v, k) for k, v in d.items()]
        items.sort()
        items.reverse()
        itemsOut = [(k, v) for v, k in items]

        highest = itemsOut[startindex:startindex + length]
        dd = dict(highest)
        wanted_keys = dd.keys()
        dictshow = dict((k, d[k]) for k in wanted_keys if k in d)

        return dictshow


dictshow = SequenceSelection(dictionary=dict2, length=7, startindex=0)

print(dictshow)


# Plot most frequent words
n = range(len(dictshow))
plt.bar(n, dictshow.values(), align='center')
plt.xticks(n, dictshow.keys())
plt.title("Most frequent Words")
plt.savefig("FrequentWords.png", transparent=True)

# the plot does not work here on vsc


root_path = os.getcwd()
# Read the whole text.
with open(path.join(root_path, 'Output.txt'), 'r', encoding='utf-8', errors='ignore') as outout_file:
    text = outout_file.readlines()


# Optional additional stopwords
stopwords = set(STOPWORDS)
stopwords.add("said")

# Construct Word Cloud
# no backgroundcolor and mode = 'RGBA' create transparency
wc = WordCloud(max_words=1000,
               stopwords=stopwords, mode='RGBA', background_color=None)

# Pass Text
wc.generate(text[0])

# store to file
wc.to_file(path.join(root_path, "jfk_moon.png"))

# show
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

plt.show()


sentiment = TextBlob(cleantext)
print("Score: ", sentiment.sentiment.polarity)  # Result = 1.0

"""
A ratio in sentiment analysis is a score that looks at how negative sentiment comments and positive sentiment comments are represented. Generally, 
this is represented on a scale of -1 to 1, with the low end of the scale indicating negative responses and the high end of the scale indicating positive responses.
"""

# https://www.analyticsvidhya.com/blog/2021/12/different-methods-for-calculating-sentiment-score-of-text/


def orange_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(24, 99%%, %d%%)" % random.randint(40, 70)


elon_mask = np.array(Image.open(path.join(
    root_path, r"C:\Users\bianc\OneDrive\Documents\CTI\NB\books - pdfs\js\img3.jpg")))


wc = WordCloud(background_color="white", mask=elon_mask,
               random_state=5, max_words=2000).generate(text[0])
plt.imshow(wc.recolor(color_func=orange_color_func, random_state=5))
plt.axis("off")
wc.to_file("elonmusk.png")