import numpy as np
import cv2

from deepface import DeepFace

MODEL = "Facenet512"


def _tmp(img):
    """Convert RGB image to BGR for OpenCV/DeepFace."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def run(image, references):
    """
    Run DeepFace emotion analysis and face verification.

    Age and gender analysis are intentionally removed to avoid
    unnecessary large model downloads during deployment.
    """

    # Analyze the uploaded image.
    # Only emotion analysis is used.
    analysis = DeepFace.analyze(
        _tmp(image),
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=True
    )

    # DeepFace may return a list or a dictionary.
    if isinstance(analysis, list):
        analysis = analysis[0]

    # Get detected face region.
    region = analysis.get("region", {})

    x, y, w, h = [
        int(region.get(k, 0))
        for k in ("x", "y", "w", "h")
    ]

    # Crop detected face.
    crop = image[
        max(0, y):max(0, y + h),
        max(0, x):max(0, x + w)
    ]

    # If cropping fails, use the original image.
    if crop.size == 0:
        crop = image

    # Compare uploaded face with reference faces.
    matches = []

    for name, ref in references:
        try:
            result = DeepFace.verify(
                _tmp(image),
                _tmp(ref),
                model_name=MODEL,
                detector_backend="opencv",
                enforce_detection=True
            )

            matches.append({
                "Reference": name,
                "Verified": bool(result["verified"]),
                "Distance": round(float(result["distance"]), 4),
                "Threshold": round(float(result["threshold"]), 4)
            })

        except Exception:
            # Skip reference images that cannot be processed.
            continue

    # Sort by smallest distance.
    matches.sort(key=lambda z: z["Distance"])

    # Create result message.
    if matches:
        best = matches[0]

        message = (
            f"Closest reference: {best['Reference']} — "
            f"{'verified' if best['Verified'] else 'not verified'}."
        )
    else:
        message = (
            "DeepFace analysis completed. "
            "Add reference faces for identity verification."
        )

    return {
        "ok": True,
        "image": image,
        "face": crop,
        "model": MODEL,
        "matches": matches,
        "attributes": {
            "emotion": analysis.get("emotion")
        },
        "message": message,
        "summary": message
    }