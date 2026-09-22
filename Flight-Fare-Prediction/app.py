from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.FlightPricePrediction.pipeline.Prediction_pipeline import CustomData, PredictPipeline

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory='static'),
    name='static'
)

templates = Jinja2Templates(directory='templates')

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse(
        'form.html',
        {"request":request}
    )
@app.post("/",response_class=HTMLResponse)
async def predict_datapoint(
    request:Request,
    Airline: str= Form(...),
    Source: str= Form(...),
    Destination: str = Form(...),
    Journey_Day: int = Form(...),
    Journey_Month: int = Form(...),
    Dep_Hour: int = Form(...),
    Dep_Minute: int = Form(...)
):
    data = CustomData(
        Airline=Airline,
        Source=Source,
        Destination=Destination,
        Journey_Day=Journey_Day,
        Journey_Month=Journey_Month,
        Dep_Hour=Dep_Hour,
        Dep_Minute=Dep_Minute
        
    )
    #convert input data to DataFrame
    final_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_data)
    result = round(pred[0],2)
    return templates.TemplateResponse(
            "result.html",
            {
                "request": request, "final_result": result
            }
        )
    
if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8080 )