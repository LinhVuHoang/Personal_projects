from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from src.Airbnb.pipelines.Prediction_pipeline import (
    CustomData,
    PredictPipeline
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()


# Static files
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# Jinja2 templates
templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ============================================================
# Home
# ============================================================

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# Prediction
# ============================================================

@app.post("/")
def predict(
    request: Request,
    property_type: str = Form(...),
    room_type: str = Form(...),
    bedrooms: str = Form(...),
    beds: str = Form(...),
    amenities: str = Form(...),
    accommodates: str = Form(...),
    bathrooms: str = Form(...),
    bed_type: str = Form(...),
    cancellation_policy: str = Form(...),
    cleaning_fee: str = Form(...),
    city: str = Form(...),
    host_has_profile_pic: str = Form(...),
    host_identity_verified: str = Form(...),
    host_response_rate: str = Form(...),
    instant_bookable: str = Form(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    number_of_reviews: str = Form(...),
    review_scores_rating: str = Form(...)
):
    try:

        # ----------------------------------------------------
        # Create CustomData object
        # ----------------------------------------------------

        data = CustomData(
            property_type=property_type,
            room_type=room_type,
            amenities=amenities,
            accommodates=accommodates,
            bathrooms=bathrooms,
            bed_type=bed_type,
            cancellation_policy=cancellation_policy,
            cleaning_fee=cleaning_fee =="True",
            city=city,
            host_has_profile_pic=host_has_profile_pic,
            host_identity_verified=host_identity_verified,
            host_response_rate=host_response_rate,
            instant_bookable=instant_bookable,
            latitude=latitude,
            longitude=longitude,
            number_of_reviews=number_of_reviews,
            review_scores_rating=review_scores_rating,
            bedrooms=bedrooms,
            beds=beds
        )

        # ----------------------------------------------------
        # Convert input to DataFrame
        # ----------------------------------------------------

        final_data = data.get_data_as_dataframe()
        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)
        
        result = round(float(pred[0]), 2)
        print(result)
        # ----------------------------------------------------
        # Return result to HTML
        # ----------------------------------------------------

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": result
            }
        )

    except Exception as e:

        error_message = f"Error during prediction: {str(e)}"

        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": error_message
            }
        )