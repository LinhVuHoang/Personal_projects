from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.DiamondPricePrediction.pipelines.Prediction_Pipeline import CustomData, PredictionPipeline


app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse(
        "form.html",
        {"request": request}
    )
    
@app.post("/", response_class=HTMLResponse)
async def predict_datapoint(
    request:Request,
    carat: float = Form(...),
    depth: float = Form(...),
    table: float = Form(...),
    x: float = Form(...), 
    y: float = Form(...), 
    z: float = Form(...), 
    cut: str = Form(...), 
    color: str = Form(...), 
    clarity: str = Form(...)
):
    data = CustomData(
        carat=carat,
        depth=depth,
        table=table,
        x=x,
        y=y,
        z=z,
        cut = cut,
        color=color,
        clarity=clarity
    )
    #convert input data to DataFrame
    final_data = data.get_data_as_dataframe()
    
    #make prediction
    predict_pipeline = PredictionPipeline()
    
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
    uvicorn.run( app, host="0.0.0.0", port=8080 )