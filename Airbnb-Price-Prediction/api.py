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
    propertytype: Optional[str] = Form(None),
    roomtype: Optional[str] = Form(None),
    amenties: Optional[str] = Form(None),
    accommodates: Optional[str] = Form(None),
    bathrooms: Optional[str] = Form(None),
    bedtype: Optional[str] = Form(None),
    canceltype: Optional[str] = Form(None),
    clean: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    dp: Optional[str] = Form(None),
    verify: Optional[str] = Form(None),
    hostresponse: Optional[str] = Form(None),
    instbook: Optional[str] = Form(None),
    lat: Optional[str] = Form(None),
    long: Optional[str] = Form(None),
    review: Optional[str] = Form(None),
    overallreview: Optional[str] = Form(None),
    bedrooms: Optional[str] = Form(None),
    beds: Optional[str] = Form(None)
):
    try:

        # ----------------------------------------------------
        # Create CustomData object
        # ----------------------------------------------------

        data = CustomData(
            property_type=propertytype,
            room_type=roomtype,
            amenities=amenties,
            accommodates=accommodates,
            bathrooms=bathrooms,
            bed_type=bedtype,
            cancellation_policy=canceltype,
            cleaning_fee=clean,
            city=city,
            host_has_profile_pic=dp,
            host_identity_verified=verify,
            host_response_rate=hostresponse,
            instant_bookable=instbook,
            latitude=lat,
            longitude=long,
            number_of_reviews=review,
            review_scores_rating=overallreview,
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

        # ----------------------------------------------------
        # Return result to HTML
        # ----------------------------------------------------

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "final_result": result
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