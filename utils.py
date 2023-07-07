import numpy as np
import pickle,json
import config

class Titanic_dataset():
    def __init__(self,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked):
        self.Pclass = Pclass
        self.Gender = Gender
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked

    def __Load_data(self):
        with open (config.MODEL_FILE_PATH,'rb') as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH,'r') as f:
            self.json_data=json.load(f)

    def survival_predict(self):
        self.__Load_data()

        Gender = self.json_data['Gender'][self.Gender]
        Embarked = self.json_data['Embarked'][self.Embarked]

        test_array = np.zeros([1,self.model.n_features_in_])
        test_array[0,0] = self.Pclass
        test_array[0,1] = Gender
        test_array[0,2] = self.Age
        test_array[0,3] = self.SibSp
        test_array[0,4] = self.Parch
        test_array[0,5] = self.Fare
        test_array[0,6] = Embarked

        predicted_survival = self.model.predict(test_array)[0]
        return predicted_survival
            