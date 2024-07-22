from src.stockpriceprediction.pipelines.prediction_pipeline import PredictPipeline
from src.stockpriceprediction.pipelines.prediction_pipeline import CustomData


from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        obj=CustomData(
                        Open=float(request.form.get("Open")),
                        High=float(request.form.get("High")),
                        Low=float(request.form.get("Low")),
                        Adj_Close=float(request.form.get("Adj_Close")),
                        Volume=float(request.form.get("Volume")),
                        
        )
        
         # this is my final data
        final_data=obj.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.Predict(final_data)
        result=pred
        
        
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
