import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.stockpriceprediction.exception import customexception
from src.stockpriceprediction.logger import logging

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.stockpriceprediction.utils.utils import save_object


@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Data_transformation_config()
       
    def get_data_transformation(self):
        
        try:
            #Define which columns should be ordinal-encoded and which should be scaled
            
            numerical_cols=['Open', 'High', 'Low', 'Adj_Close', 'Volume']
            
           
            
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols)
           ])
            
            return preprocessor
           
        except Exception as e:
            logging.info("Exception occured in initiating data transformation")
            raise customexception(e,sys)
        
    def initiated_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data completed")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj=self.get_data_transformation()
            
            # Sample DataFrame
            train_df = pd.DataFrame(train_df)
            test_df=pd.DataFrame(train_df)
            
            

            # Sample DataFrame
            train_df = pd.DataFrame(train_df)

            # Calculate IQR
            Q1 = train_df['Volume'].quantile(0.25)
            Q3 = train_df['Volume'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = train_df[(train_df['Volume'] < lower_bound) | (train_df['Volume'] > upper_bound)]

            # Calculate the mean of non-outliers
            mean_value = train_df[(train_df['Volume'] >= lower_bound) & (train_df['Volume'] <= upper_bound)]['Volume'].mean()

            # Replace outliers with the mean value
            train_df['Volume'] = train_df['Volume'].apply(lambda x: mean_value if (x < lower_bound or x > upper_bound) else x)
            
          
           
            test_df = pd.DataFrame(test_df)

            # Calculate IQR
            Q1 = test_df['Volume'].quantile(0.25)
            Q3 = test_df['Volume'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers =  test_df[(train_df['Volume'] < lower_bound) | (test_df['Volume'] > upper_bound)]

            # Calculate the mean of non-outliers
            mean_value = test_df[(train_df['Volume'] >= lower_bound) & (test_df['Volume'] <= upper_bound)]['Volume'].mean()

            # Replace outliers with the mean value
            test_df['Volume'] = test_df['Volume'].apply(lambda x: mean_value if (x < lower_bound or x > upper_bound) else x)
           
            
            
            input_feature_train_df=train_df.rename(columns = {'Adj Close':'Adj_Close'}, inplace = True)
            input_feature_train_df = train_df.drop("Date",axis=1)
            input_feature_train_df = train_df.drop("Close",axis=1)
            target_feature_train_df=train_df["Close"]
            
            input_feature_test_df=test_df.rename(columns = {'Adj Close':'Adj_Close'}, inplace = True)
            input_feature_test_df = test_df.drop("Date",axis=1)
            input_feature_test_df = test_df.drop("Close",axis=1)
            target_feature_test_df=train_df["Close"]
            
            
            
            
            
            input_feature_train_arr=preprocessing_obj.fit_transform( input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
          
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr,
               
                )
        
        
        except Exception as e:
            logging.info("Exception occured in initiating data transformation")
            raise customexception(e,sys)
        
        