from src.stockpriceprediction.pipelines.prediction_pipeline import CustomData
from src.stockpriceprediction.pipelines.prediction_pipeline import PredictPipeline


obj=CustomData(11.09950,11.204875,10.994125,0.101497,24400)

final_data=obj.get_data_as_dataframe()
        
predict_pipeline=PredictPipeline()
        
pred=predict_pipeline.Predict(final_data)
        
result=pred

print(final_data)
print(result)