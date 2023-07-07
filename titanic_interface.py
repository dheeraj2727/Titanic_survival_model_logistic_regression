from flask import Flask,render_template,request,jsonify
import config
from utils import Titanic_dataset

app = Flask(__name__)

@app.route('/titanic_model')

def home():
    return render_template('titanic.html')

@app.route('/predict_survival',methods=['GET','POST'])

def prediction():
    if request.method=='GET':
        data = request.args.get
        Pclass = int(data('Pclass'))
        Gender = data('Gender')
        Age = int(data('Age'))
        SibSp = int(data('SibSp'))
        Parch=int(data('Parch'))
        Fare = float(data('Fare'))
        Embarked=data('Embarked')

        obj = Titanic_dataset(Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)
        status = obj.survival_predict()

        if status==0:
            pred = 'Not Survived'
        else:
            pred = 'Survived'

        return render_template('titanic.html',prediction=pred)
    
    elif request.method=='POST':
        data = request.form
        Pclass = int(data['Pclass'])
        Gender = data['Gender']
        Age = int(data['Age'])
        SibSp = int(data['SibSp'])
        Parch=int(data['Parch'])
        Fare = float(data['Fare'])
        Embarked=data['Embarked']

        obj = Titanic_dataset(Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)
        status = obj.survival_predict()

        if status==0:
            pred = 'Not Survived'
        else:
            pred = 'Survived'

        return render_template('titanic.html',prediction=pred)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=config.PORT_NUMBER)
