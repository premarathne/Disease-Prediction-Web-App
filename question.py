from flask import Flask,render_template,url_for,redirect
from flask import request
import pandas as pd
import ast

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import csv,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv('Training.csv')
#df=pd.DataFrame(data)
#cols=df.columns
#cols=cols[:-1]
#X=df[cols]
#y=df['prognosis']
#df = pd.read_csv('Training.csv')
X =df.values[:, 1:132]
y =df.values[:, 132]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dt = DecisionTreeClassifier()
clf_entropy=dt.fit(X_train,y_train)
y_pred=clf_entropy.predict(X_test)
#print(y_pred)
#pickle.dump(clf_entropy,open('model.pk1','wb'))
indices=[i for i in range(132)]
symptoms=df.columns.values[:-1]
#print(symptoms)
dictionary=dict(zip(symptoms,indices))
app=Flask(__name__)
symptom1=''
greeting = "Hiii!!! I am a Symptom Checker.I will help you to identify your disease. "
question1 = "Can you Tell me one of symptom you have????"
warn=''
@app.route('/')
def que():
	return render_template("symptom_predictor.html",greeting=greeting,que1=question1)


@app.route('/', methods=['POST','GET'])
def result():
	if request.method=='POST':


		symptom1 = request.form['symptom1']
		#symptom1='itching'
		df=pd.read_csv('Training.csv')
		df=df.loc[df[symptom1]==1]
		df=df.drop([symptom1],axis=1)
		df=df.drop(['prognosis'],axis=1)
		symptom_list=[]
		
		for i in df.columns:
  			for j in range(0,len(df.index)):
    	 	 		if df[i].iloc[j]==1:
      	   				if i not in symptom_list:
            					symptom_list.append(i)

        #return symptom_list:


		#for i in symptom_list:
  			#print(i)

		#for i in range(0,len(symptom_list)):
		item=symptom_list[0]
		#name='Anjula'
		#symptom_list2=[]
		#print(request.form.get('check'))
		#print(symptom_list2)
		return redirect(url_for('ques',lists=symptom_list))
		#return render_template('q2.html',item=item)

	#return render_template('q2.html')
@app.route('/ques/<lists>')
def ques(lists):
	print(type(lists))
	print(lists)
	#list = lists.strip('][').split(', ')
	list = ast.literal_eval(lists)
	#data=list(lists)
	#lists=lists.strip('][')
	#print(lists)
	#lists=lists.strip("','")
	print(lists)
	#print(lists)
	#data = list(lists.split("','"))
	#print(data)
	#data=string.split(lists)

	#data=lists.split(",")
	#lists = list(lists.split())
	#type(lists)
	#print(lists)
	#for i in range(0,len(data)):

	return render_template('ask_question.html',list=list)
	#if request.method=='POST':
		#for i in range(0,len(data)):
		#return redirect(url_for('ques',items=data[0]))
@app.route('/answer', methods=['GET ','POST'])
def answer():
    if request.method=='POST':

        print(request.form.getlist('check'))

        #for(i in range[1:106])
        #selected_symptoms = []
        symptom=request.form.getlist('check')
       # print(symptom)
        symptoms=[0 for i in range(131)]
        print(symptoms)
        for i in symptom:
        	#print(i)
            idx=dictionary[i]
            print(idx)
            print(i)
            symptoms[idx]=1
        symptoms=np.matrix(symptoms)
       # print(symptoms)
        symptoms=np.reshape(symptoms,-1)
       # print(symptoms)
        model_prediction=clf_entropy.predict(symptoms)
        #print(model_prediction)



        #ml_model=joblib.load(loaded_model)
        #result = ml_model.predict(selected_symptoms)

    return render_template('result.html',result=model_prediction,symptoms=symptoms)
if __name__ == '__main__' :
    app.run(debug=True)
