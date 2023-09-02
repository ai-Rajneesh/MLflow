import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def get_data():
    #URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        df = pd.read_csv('/Users/rajneeshyadav/Desktop/Raj4.0/FSDS 2.0/MLflow/notebooks/data/gemstone.csv')
        return df
    except Exception as e:
        raise e
    
def evaluate(y_true,y_pred):
    mae=mean_absolute_error(y_true, y_pred)
    mse=mean_squared_error(y_true, y_pred)
    rmse=np.sqrt(mean_squared_error(y_true, y_pred))
    r2=r2_score(y_true, y_pred)   
    
    return mae,mse,rmse,r2
    
    
    

def main():
    df = get_data() 
    
    df=df.drop(labels=['id'],axis=1)
    
    # Independent and dependent features
    X = df.drop(labels=['price'],axis=1)
    y = df[['price']]
    
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns
    
    # Define the custom ranking for each ordinal variable
    cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
    color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
    
    num_pipeline=Pipeline(
        steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
        ]
    )

    cat_pipeline=Pipeline(
        steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
        ('scaler',StandardScaler())
        ]
    )

    preprocessor=ColumnTransformer([
        ('num_pipeline',num_pipeline,numerical_cols),
        ('cat_pipeline',cat_pipeline,categorical_cols)
    ])
    
    # Split data before preprocessing
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=51)
    
    # Apply preprocessing to the split data
    X_train_preprocessed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
    X_test_preprocessed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())
    
    with mlflow.start_run():
        regression=LinearRegression()
        regression.fit(X_train_preprocessed, y_train)
        pred=regression.predict(X_test_preprocessed)
        
        '''model2 = ElasticNet()
        model2.fit(X_train_preprocessed, y_train)
        pred=model2.predict(X_test_preprocessed)'''
        
        
        
        
    
    
        #evalute the model
        mae,mse,rmse,r2=evaluate(y_test,pred)
    
        mlflow.log_metric('r2_score',r2)
        
        #mlflow model logging
        mlflow.sklearn.log_model(regression,"LinearRegressionModel")
        
    
        print(f"mean absolute error {mae}, mean squared error {mse}, root mean squared error {rmse}, r2_score {r2}")
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
