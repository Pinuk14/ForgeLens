"""Metadata extraction and analysis preprocessor for image forensics."""

from datetime import datetime
from io import BytesIO
from PIL import Image


def extract_metadata(image_bytes: bytes) -> dict:
    """Extract metadata (EXIF) from raw image bytes and return a feature dict.

    This function parses EXIF data using Pillow to find tags for device,
    software, timestamps, and GPS existence. It computes the delta between
    creation and digitization timestamps, identifies known editing software signatures,
    and constructs a normalized feature vector for downstream fusion models.

    All parsing is wrapped in try/except blocks to ensure it never raises
    an exception, returning fallback default values on failure or missing metadata.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        A dictionary containing parsed metadata fields and a normalized feature_vector.
    """
    # Default return structure
    result = {
        "has_exif": False,
        "make": None,
        "model": None,
        "software": None,
        "gps_present": False,
        "datetime_original": None,
        "datetime_digitized": None,
        "datetime_delta_hours": None,
        "suspicious_software": False,
        "feature_vector": [0.0] * 16
    }

    try:
        img = Image.open(BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return result

        # Helper to clean and decode exif values to string
        def clean_val(v) -> str | None:
            if v is None:
                return None
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8", errors="ignore")
                except Exception:
                    return None
            if isinstance(v, str):
                return v.strip().replace("\x00", "")
            return str(v)

        # Standard Exif Tag IDs:
        # Make: 271 (0x010f), Model: 272 (0x0110), Software: 305 (0x0131)
        make = clean_val(exif.get(271))
        model = clean_val(exif.get(272))
        software = clean_val(exif.get(305))

        # Check Exif SubIFD (34665) for timestamps
        # DateTimeOriginal: 36867 (0x9003), DateTimeDigitized: 36868 (0x9004)
        exif_sub = exif.get_ifd(34665)
        dt_orig = clean_val(exif.get(36867) or exif_sub.get(36867))
        dt_dig = clean_val(exif.get(36868) or exif_sub.get(36868))

        # Check GPS Info IFD (34853)
        gps_info = exif.get_ifd(34853)
        gps_present = bool(gps_info)

        # Determine if EXIF data is present (if we found any expected fields)
        has_exif = bool(make or model or software or dt_orig or dt_dig or gps_present or len(exif) > 0)

        # Calculate time delta between original and digitized timestamps
        datetime_delta_hours = None
        if dt_orig and dt_dig:
            formats = [
                "%Y:%m:%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S"
            ]
            dt_o_obj = None
            dt_d_obj = None
            for fmt in formats:
                if not dt_o_obj:
                    try:
                        dt_o_obj = datetime.strptime(dt_orig, fmt)
                    except ValueError:
                        pass
                if not dt_d_obj:
                    try:
                        dt_d_obj = datetime.strptime(dt_dig, fmt)
                    except ValueError:
                        pass

            if dt_o_obj and dt_d_obj:
                delta_seconds = abs((dt_o_obj - dt_d_obj).total_seconds())
                datetime_delta_hours = delta_seconds / 3600.0

        # Check for suspicious editing software
        suspicious_software = False
        if software:
            sw_lower = software.lower()
            keywords = [
                "photoshop", "lightroom", "gimp", "affinity", "canva",
                "dall-e", "midjourney", "stable diffusion", "firefly", "adobe"
            ]
            if any(kw in sw_lower for kw in keywords):
                suspicious_software = True

        # Construct the feature vector
        # [0] has_exif as float
        # [1] suspicious_software as float
        # [2] gps_present as float
        # [3] make present as float
        # [4] model present as float
        # [5] datetime_delta_hours normalized to [0,1] with max=8760 (1 year)
        # [6-15] zeros (reserved for JPEG quantization features in future)
        fv = [0.0] * 16
        fv[0] = 1.0 if has_exif else 0.0
        fv[1] = 1.0 if suspicious_software else 0.0
        fv[2] = 1.0 if gps_present else 0.0
        fv[3] = 1.0 if make is not None else 0.0
        fv[4] = 1.0 if model is not None else 0.0
        if datetime_delta_hours is not None:
            fv[5] = min(datetime_delta_hours / 8760.0, 1.0)
        else:
            fv[5] = 0.0

        # Populate the dictionary
        result["has_exif"] = has_exif
        result["make"] = make
        result["model"] = model
        result["software"] = software
        result["gps_present"] = gps_present
        result["datetime_original"] = dt_orig
        result["datetime_digitized"] = dt_dig
        result["datetime_delta_hours"] = datetime_delta_hours
        result["suspicious_software"] = suspicious_software
        result["feature_vector"] = fv

    except Exception:
        # Fall back to safe defaults in case of any processing failure
        pass

    return result
