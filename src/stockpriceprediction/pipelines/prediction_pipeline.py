import os
import sys
import pandas as pd
from src.stockpriceprediction.exception import customexception
from src.stockpriceprediction.logger import logging
from src.stockpriceprediction.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def Predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            Preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=Preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
    
    
        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    
    def __init__(self,Open:float,
                    High:float,
                    Low:float,
                    Adj_Close:float,
                    Volume :float,
                    ):
        
        self.Open=Open
        self.High=High
        self.Low=Low
        self.Adj_Close= Adj_Close
        self.Volume =Volume 
       
    
    def get_data_as_dataframe(self):
        
        try:
            custom_data_input_dict={
                'Open':[self.Open],
                'High':[self.High],
                'Low':[self.Low],
                'Adj_Close':[self.Adj_Close],
                'Volume':[self.Volume],
                
                }
                
                
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
        
        
            