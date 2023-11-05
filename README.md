# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect a labeled dataset of emails, distinguishing between spam and non-spam.
2. Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.
3. Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.
4. Split the dataset into a training set and a test set.
5. Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.
6. Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.
7.  Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.
8. Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.
9. Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```

Program to implement the SVM For Spam Mail Detection
Developed by: SHALINI VENKATESAN
RegisterNumber:  212222240096

```
```
import pandas as pd
data = pd.read_csv("spam.csv",encoding = "windows - 1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### data.head()

![image](https://github.com/JoyceBeulah/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343698/bb4cb93e-ffe0-42e1-b094-d551c018c21c)

### data.info()

![image](https://github.com/JoyceBeulah/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343698/10f79e94-b950-4eab-adf1-3e0bbab6916c)

### data.isnull().sum()

![image](https://github.com/JoyceBeulah/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343698/b4d6aa56-c2d8-4046-b3ce-1fa79e9ab443)

### Y_prediction value

![image](https://github.com/JoyceBeulah/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343698/9524331f-f492-4b40-a4f2-d67ecc99bcbc)

### Accuracy value

![image](https://github.com/JoyceBeulah/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343698/e4210c7a-c428-4a6e-aaa5-b910a0df6f2f)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
