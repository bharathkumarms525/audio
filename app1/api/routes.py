from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from app1.services.transcription import transcribe_audio_file
from app1.services.all_graphs import plot_audio_with_speakers_akshat, plot_audio_with_speakers_bharath
from app1.utils.logging_config import logging
import tempfile
import os

router = APIRouter()

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        response = await transcribe_audio_file(file)
        return JSONResponse(content=response)
    except Exception as e:
        logging.error(f"Error in /transcribe/ endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred during transcription."}, status_code=500)

@router.post("/plot/akshat/")
async def plot_akshat(file: UploadFile = File(...), sr: int = Query(10), threshold: int = Query(90)):
    """
    API endpoint to generate Akshat's plot.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        plot_audio_with_speakers_akshat(temp_file_path, sr=sr, threshold=threshold)
        return JSONResponse(content={"message": "Akshat's plot generated successfully."})
    except Exception as e:
        logging.error(f"Error in /plot/akshat/ endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred while generating Akshat's plot."}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/plot/bharath/")
async def plot_bharath(
    file: UploadFile = File(...),
    lufs_threshold_value: float = Query(default=None)
):
    """
    API endpoint to generate Bharath's plot with optional LUFS threshold.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        await plot_audio_with_speakers_bharath(
            temp_file_path,
            lufs_threshold_value=lufs_threshold_value
        )
        return JSONResponse(content={"message": "Bharath's plot generated successfully."})
    except Exception as e:
        logging.error(f"Error in /plot/bharath/ endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred while generating Bharath's plot."}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

