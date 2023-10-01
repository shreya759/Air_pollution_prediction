from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import pickle
import sqlite3

app = Flask(__name__)

price=pd.read_csv("processeddata.csv")
df = pd.read_csv('processeddata.csv')

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['state']= label_encoder.fit_transform(df['state'])
df['location']= label_encoder.fit_transform(df['location'])
df['type']= label_encoder.fit_transform(df['type'])
df['state'].unique()
df['location'].unique()
df['type'].unique()

X=df[['state','type','SOi','Noi','Rpi','SPMi']]
Y=df['AQI']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix

DT=DecisionTreeRegressor()
DT.fit(X_train,Y_train)
train_preds=DT.predict(X_train)
#predicting on test
test_preds=DT.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',DT.score(X_train, Y_train))
print('RSquared value on test:',DT.score(X_test, Y_test))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("intro.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("intro.html")
    else:
        return render_template("signup.html")



@app.route('/index')
def index():
    companies=sorted(price['state'].unique())
    sellingprice=sorted(price['location'].unique(),reverse=True)
    fuel=sorted(price['type'].unique())
    

    companies.insert(0,"Select State ")
    sellingprice.insert(0, "Select Location")
    fuel.insert(0, "Select Area Type ")
    

    return render_template("index.html",companies=companies,sellingprice=sellingprice,fuel=fuel)


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        
        text1=request.form.get('1')
        if text1 == 'Maharashtra':
            state = 19
        elif text1 == 'Uttar Pradesh':
            state = 35
        elif  text1 == 'Andhra Pradesh':
            state = 0
        elif text1== 'Punjab':
            state = 27
        elif text1=='Rajasthan': 
            state = 28
        elif text1=='Kerala':
            state = 17
        elif text1=='Himachal Pradesh':
            state = 13
        elif text1=='West Bengal':
            state = 36
        elif text1 =='Gujarat':
            state = 11
        elif text1 =='Tamil Nadu':
            state = 30
        elif text1 =='Madhya Pradesh':
            state = 20
        elif text1 =='Assam':
            state = 3
        elif text1 =='Odisha':                         
            state = 25
        elif text1 =='Karnataka':                     
            state = 16
        elif text1 =='Delhi':
            state = 9
        elif text1 =='Chandigarh':
            state = 5
        elif text1 =='Chhattisgarh':
            state = 6
        elif text1 =='Goa':                             
            state = 10
        elif text1 =='Jharkhand':                       
            state = 15
        elif text1 =='Mizoram':                        
            state = 23
        elif text1 =='Telangana':                       
            state = 31
        elif text1 =='Meghalaya':                       
            state = 22
        elif text1 =='Puducherry':
            state = 26
        elif text1 =='Haryana':
            state = 11
        elif text1 =='Nagaland':
            state = 24
        elif text1 =='Bihar':
            state = 4
        elif text1 =='Uttarakhand':
            state = 34
        elif text1 =='Jammu & Kashmir':
            state = 14
        elif text1 =='Daman & Diu':
            state = 8
        elif text1 =='Dadra & Nagar Haveli':
            state = 7
        elif text1 =='Uttaranchal':
            state = 33
        elif text1 =='Arunachal Pradesh':
            state = 1
        elif text1 =='Manipur':
            state = 21
        elif text1 =='Lakshadweep':
            state = 18
        elif text1 =='andaman-and-nicobar-islands':
            state = 2
        elif text1 =='Sikkim':
            state = 29
        elif text1 =='Tripura':
            state = 32
        text2=request.form.get('2')
        text3=request.form.get('3')
        if text3 == 'Residential, Rural and other Areas':
            area = 6
        elif text3 == 'Industrial Area':
            area = 1
        elif text3 == 'Residential and others':
            area = 8
        elif text3 == 'Industrial Areas':
            area = 2
        elif text3 == 'Sensitive Area':
            area = 5
        elif text3 == 'Sensitive Areas':
            area = 9
        elif text3 == 'RIRUO':
            area = 0
        elif text3 == 'Sensitive':
            area = 4
        elif text3 == 'Industrial':
            area = 3
        elif text3 == 'Residential':
            area = 7
        text4=request.form.get('4')
        text5=request.form.get('5')
        text6=request.form.get('6')
        text7=request.form.get('7')

        lr_pm = DT.predict([[state, area, text4,text5,text6,text7]])

        print(lr_pm)

    return render_template("result.html", lr_pm = np.round(lr_pm,3))

if __name__ == "__main__":
    app.run(debug = True)
