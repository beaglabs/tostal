"""Temporal activity: ML model inference on GPU."""

from temporalio import activity


@activity.defn
async def run_inference(params: dict) -> dict:
    task = params["task"]
    model_type = params["model_type"]
    data = params["data"]
    parameters = params.get("parameters", {})

    # Stub: in production, loads model from container-models store
    # and runs PyTorch inference on the input data
    result_data = data

    return {
        "data": result_data,
        "shape": [],  # populated by actual inference
        "classes": [],
        "confidence": 0.0,
    }