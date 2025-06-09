from io import BytesIO
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, Json

app = FastAPI()

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Point(BaseModel):
    """A 2D point."""

    x: float
    y: float


class OverlaySpec(BaseModel):
    points1: list[Point] = Field(..., min_length=3, max_length=3)
    points2: list[Point] = Field(..., min_length=3, max_length=3)


def pad(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pad an image with a white border that's symmetric."""
    top = (H - img.shape[0]) // 2
    bottom = H - img.shape[0] - top
    left = (W - img.shape[1]) // 2
    right = W - img.shape[1] - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255])


def pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad two images to same size."""
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    return pad(a, H, W), pad(b, H, W)


@app.post("/api/overlay")
@app.post("/")
async def overlay(
    request: Annotated[Json[OverlaySpec], Form(...)],
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
) -> StreamingResponse:
    ptsA = np.array([[p.x, p.y] for p in request.points1], dtype=np.float32)
    ptsB = np.array([[p.x, p.y] for p in request.points2], dtype=np.float32)
    M = cv2.getAffineTransform(ptsB, ptsA)

    dataA = await image1.read()
    A = cv2.imdecode(np.frombuffer(dataA, np.uint8), cv2.IMREAD_GRAYSCALE)
    dataB = await image2.read()
    B = cv2.imdecode(np.frombuffer(dataB, np.uint8), cv2.IMREAD_GRAYSCALE)

    B = cv2.warpAffine(
        B, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=[255]
    )

    A, B = pad_to_match(A, B)
    A = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
    B = np.stack([np.full_like(B, 255), B, B], axis=2)

    overlay_img = cv2.addWeighted(A, 0.5, B, 0.5, 0)
    ret, jpeg = cv2.imencode(".jpg", overlay_img)
    buffer = BytesIO(jpeg.tobytes())
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
