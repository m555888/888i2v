"""
Shared generation logic for Image-to-Video (used by Streamlit app and Telegram bot).
No Streamlit imports.
"""
import os
import json
import requests
from pathlib import Path
from datetime import datetime

import fal_client
import replicate

def _get_data_root() -> Path:
    root = Path(__file__).resolve().parent
    try:
        test = root / ".write_check"
        test.write_text("")
        test.unlink(missing_ok=True)
        return root
    except (OSError, PermissionError):
        tmp = Path("/tmp") / "imagetovideo_app"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

DATA_ROOT = _get_data_root()
VIDEO_DIR = DATA_ROOT / "generated_videos"
CONFIG_FILE = DATA_ROOT / "config.json"

HOMOGLYPH = str.maketrans({
    "a": "ɑ", "e": "е", "i": "і", "o": "о", "s": "ѕ", "c": "с", "p": "р",
    "x": "х", "y": "у", "A": "А", "E": "Е", "O": "О", "B": "В", "H": "Н",
})


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(
    key_id: str = "",
    key_secret: str = "",
    api_key: str = "",
    model: str = "",
    replicate_api_token: str = "",
):
    data = load_config()
    data["key_id"] = (key_id or "").strip()
    data["key_secret"] = (key_secret or "").strip()
    data["api_key"] = (api_key or "").strip()
    data["model"] = model
    data["replicate_api_token"] = (replicate_api_token or "").strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# پرامپت پیش‌فرض که به همهٔ پرامپت‌ها اضافه می‌شود (طبیعی‌بودن بدن، عدم حرکت خلاف فیزیک، فقط روی شخص در عکس)
DEFAULT_PROMPT_SUFFIX = (
    "Natural body proportions, no weird or physics-defying movements, "
    "everything natural and human-like. Apply only to the person (woman) in the image; prompt refers to the female subject in the photo."
)


def obfuscate_prompt(text: str) -> str:
    return text.translate(HOMOGLYPH)


# ترجمهٔ فارسی / فینگلیش به انگلیسی برای پرامپت ویدئو.
# ابتدا اگر متن فارسی‌اسکریپتی باشد از LibreTranslate استفاده می‌کنیم؛
# در غیر این صورت (یا اگر خروجی عوض نشد)، از LLM فال استفاده می‌کنیم تا
# هم فارسی و هم فینگلیش را به یک پرامپت انگلیسی طبیعی تبدیل کند.
LIBRETRANSLATE_URL = "https://libretranslate.com/translate"


def _looks_persian_script(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text or "")


def translate_to_english(text: str, timeout: int = 10) -> str:
    """ترجمهٔ متن به انگلیسی (فارسی یا فینگلیش). در صورت خطا همان متن اصلی برمی‌گردد."""
    text = (text or "").strip()
    if not text:
        return text
    original = text

    # 1) اگر حروف فارسی دارد، اول LibreTranslate را امتحان کن
    if _looks_persian_script(text):
        try:
            r = requests.post(
                LIBRETRANSLATE_URL,
                json={"q": text, "source": "auto", "target": "en", "format": "text"},
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if r.ok:
                out = r.json().get("translatedText")
                if out and isinstance(out, str) and out.strip() and out.strip() != original:
                    return out.strip()
        except Exception:
            pass

    # 2) fallback: استفاده از LLM فال (handles Farsi + Fingilish, or leave good English as-is)
    try:
        cfg = load_config()
        kid = (cfg.get("key_id") or "").strip()
        ksec = (cfg.get("key_secret") or "").strip()
        raw_api = (cfg.get("api_key") or "").strip()
        api_key = f"{kid}:{ksec}" if (kid and ksec) else (raw_api if ":" in raw_api else raw_api)
        if not api_key:
            api_key = os.environ.get("FAL_KEY", "").strip()
        if api_key:
            system = (
                "You receive a user text that may be:\n"
                "- Persian in Arabic script\n"
                "- Persian written in Latin letters (Fingilish)\n"
                "- or already an English image-to-video prompt.\n\n"
                "Your job:\n"
                "1) If the text is Persian (script or Fingilish), first understand it fully and then produce a SINGLE, clear English video-generation prompt.\n"
                "2) The English prompt must be friendly for image-to-video models and VERY DETAILED, but you should NOT invent new lighting; assume lighting and overall look stay consistent with the reference image unless the user explicitly asks otherwise.\n"
                "   - Describe the SCENE and environment (time of day, place, atmosphere) in a way that is compatible with the user's image.\n"
                "   - Describe the MAIN SUBJECT(S) with appearance, clothing, facial expression, and emotional state.\n"
                "   - Describe FULL BODY posture and motion over time (how they move, walk, turn, shift weight, gesture, look around, interact with objects, etc.).\n"
                "   - Think explicitly about how a real human body would move (joints, balance, gravity) and phrase motions so they are smooth and physically believable.\n"
                "   - Describe CAMERA framing and movement (starting shot type and angle, tracking, dolly, pan, tilt, zoom, orbit, cuts if any) so the model understands where the camera is and how it moves.\n"
                "3) Motions and physics must feel natural and realistic. Avoid anything impossible, jittery, or glitchy; avoid sudden teleporting or broken limbs.\n"
                "4) You may add helpful cinematic details (camera moves, smooth transitions, subtle secondary motions like hair, clothes, environment reacting) BUT they must stay consistent with what the user wants and must NOT change the story or add new characters.\n"
                "5) If the input is already a strong English video prompt, you may enrich it with more body-motion and camera/lighting detail, but keep the same core idea and style.\n"
                "6) The final answer must be 1–3 concise sentences, no bullet points, no explanations, no quotes.\n"
                "7) NEVER mention that you are translating or enhancing; just output the final English prompt."
            )
            client = fal_client.SyncClient(key=api_key)
            out = client.subscribe(
                "fal-ai/any-llm",
                arguments={
                    "prompt": text,
                    "system_prompt": system,
                    "model": "google/gemini-2.5-flash-lite",
                    "max_tokens": 300,
                },
            )
            if out is not None:
                if isinstance(out, str):
                    res = out.strip()
                elif isinstance(out, dict):
                    data = out.get("data") or out
                    res = data.get("output") or data.get("text") or data.get("result")
                    if isinstance(res, list) and res and isinstance(res[0], dict):
                        res = res[0].get("content") or res[0].get("text")
                    if not isinstance(res, str):
                        res = ""
                else:
                    res = ""
                if res and res.strip():
                    return res.strip()
    except Exception:
        pass

    # اگر همه‌چیز شکست خورد، همان متن اولیه را برگردان
    return original


MODELS = {
    "Seedance 2": {
        "provider": "fal",
        "id": "fal-ai/bytedance/seedance/v2/image-to-video",  # روی fal «Coming soon»؛ وقتی لایو شد همین کلید کار می‌کند
        "duration_map": {5: 5, 10: 10, 15: 15},
        "image_param": "image_url",
        "badge": "✿",
    },
    "Seedance 1.5 Pro": {
        "provider": "fal",
        "id": "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
        "duration_map": {5: 5, 10: 10, 15: 12},
        "image_param": "image_url",
        "badge": "✿",
    },
    "Seedance 1.0 Pro": {
        "provider": "fal",
        "id": "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        "duration_map": {5: 5, 10: 10, 15: 12},
        "image_param": "image_url",
        "badge": "✿",
    },
    "Seedance 1.0 Lite": {
        "provider": "fal",
        "id": "fal-ai/bytedance/seedance/v1/lite/image-to-video",
        "duration_map": {5: 5, 10: 10, 15: 12},
        "image_param": "image_url",
        "badge": "✿",
    },
    "Wan 2.1 Uncensored": {
        "provider": "replicate",
        "id": "uncensored-com/wan2.1-uncensored-video-lora:46cfc445b5f89469deb11b5d8227ff9e3bb129c8920f3886cd78c426f43204c4",
        "duration_map": {5: 81, 10: 81, 15: 81},
        "image_param": "image",
        "badge": "◉",
        "trigger_word": "unai,",
    },
}


def get_available_models(config: dict) -> list:
    out = []
    fal_ok = bool((config.get("key_id") or "").strip() and (config.get("key_secret") or "").strip()) or bool(
        (config.get("api_key") or "").strip()
    )
    rep_ok = bool((config.get("replicate_api_token") or "").strip())
    for name, cfg in MODELS.items():
        p = cfg.get("provider", "fal")
        if p == "fal" and fal_ok:
            out.append(name)
        elif p == "replicate" and rep_ok:
            out.append(name)
    return out


def save_video_from_url(video_url: str) -> str | None:
    try:
        VIDEO_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = VIDEO_DIR / f"video_{ts}.mp4"
        r = requests.get(video_url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        return str(path)
    except Exception:
        return None


def get_aspect_from_path(path) -> str:
    if not path or not Path(path).exists():
        return "16:9"
    try:
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path))
        w, h = img.size
        r = w / h if h else 1
        if 0.9 < r < 1.1:
            return "1:1"
        return "16:9" if r > 1 else "9:16"
    except Exception:
        return "16:9"


def generate_video_fal(
    model_config: dict,
    image_url: str,
    prompt: str,
    user_duration: int,
    aspect_ratio: str,
    api_key: str = "",
    use_obfuscation: bool = True,
) -> dict:
    duration = model_config["duration_map"].get(user_duration, user_duration)
    img_param = model_config.get("image_param", "image_url")
    api_prompt = obfuscate_prompt(prompt) if use_obfuscation else prompt
    params = {"prompt": api_prompt, img_param: image_url}
    params["duration"] = str(duration) if ("kling" in model_config["id"].lower() or "seedance" in model_config["id"].lower()) else duration
    params["aspect_ratio"] = aspect_ratio
    if "seedance" in model_config["id"].lower():
        params["generate_audio"] = False
        params["enable_safety_checker"] = False
    client = fal_client.SyncClient(key=api_key.strip() if api_key else None)
    return client.subscribe(model_config["id"], arguments=params, with_logs=False)


def generate_video_replicate(
    model_config: dict,
    image_source,
    prompt: str,
    user_duration: int,
    aspect_ratio: str,
    api_token: str,
    use_obfuscation: bool = True,
) -> dict:
    api_prompt = obfuscate_prompt(prompt) if use_obfuscation else prompt
    trigger = model_config.get("trigger_word", "")
    if trigger and not api_prompt.startswith(trigger):
        api_prompt = trigger + " " + api_prompt
    model_id = model_config["id"]
    img_key = model_config.get("image_param", "image")
    inp = {img_key: image_source, "prompt": api_prompt}
    frames = model_config["duration_map"].get(user_duration)
    if frames:
        inp["frames"] = frames
    if aspect_ratio in ("16:9", "9:16", "1:1"):
        inp["aspect_ratio"] = aspect_ratio
    os.environ["REPLICATE_API_TOKEN"] = api_token
    out = replicate.run(model_id, input=inp)
    if hasattr(out, "url"):
        return {"video": {"url": out.url}, "url": out.url}
    if isinstance(out, str):
        return {"video": {"url": out}, "url": out}
    if isinstance(out, (list, tuple)) and len(out):
        u = out[0]
        return {"video": {"url": getattr(u, "url", None) or str(u)}, "url": getattr(u, "url", None) or str(u)}
    if isinstance(out, dict) and out.get("url"):
        return {"video": {"url": out["url"]}, "url": out["url"]}
    return out or {}


def run_one_generation(
    image_path: str,
    prompt: str,
    model_name: str,
    duration: int = 5,
    aspect_ratio: str | None = None,
    config: dict | None = None,
) -> tuple[str, str]:
    """
    Run image-to-video for one job. Returns (video_url, local_path).
    Raises on error.
    """
    from PIL import Image, ImageOps
    config = config or load_config()
    for env_key, cfg_key in (
        ("FAL_KEY", "api_key"),
        ("REPLICATE_API_TOKEN", "replicate_api_token"),
    ):
        if not (config.get(cfg_key) or "").strip() and os.environ.get(env_key):
            config[cfg_key] = os.environ.get(env_key, "").strip()
    kid = (config.get("key_id") or "").strip()
    ksec = (config.get("key_secret") or "").strip()
    raw_api = (config.get("api_key") or "").strip()
    api_key = f"{kid}:{ksec}" if (kid and ksec) else (raw_api if ":" in raw_api else raw_api)
    rep_token = (config.get("replicate_api_token") or "").strip()
    if api_key:
        os.environ["FAL_KEY"] = api_key
    if kid:
        os.environ["FAL_KEY_ID"] = kid
    if ksec:
        os.environ["FAL_KEY_SECRET"] = ksec
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    prompt = (prompt or "").strip()
    if prompt:
        prompt = prompt + " " + DEFAULT_PROMPT_SUFFIX
    else:
        prompt = DEFAULT_PROMPT_SUFFIX
    model_config = MODELS[model_name]
    provider = model_config.get("provider", "fal")
    if aspect_ratio is None:
        aspect_ratio = get_aspect_from_path(image_path)
    use_obfuscation = True
    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_url = None
    if provider == "fal":
        if not api_key:
            raise ValueError("Fal API key not set")
        image_url = fal_client.upload_file(str(input_path))
        if not image_url:
            raise RuntimeError("Fal upload failed")
    elif provider == "replicate":
        if not rep_token:
            raise ValueError("Replicate API token not set")
        if api_key:
            try:
                image_url = fal_client.upload_file(str(input_path))
            except Exception:
                image_url = None

    result = None
    if provider == "fal":
        result = generate_video_fal(
            model_config, image_url, prompt, duration, aspect_ratio,
            api_key=api_key, use_obfuscation=use_obfuscation,
        )
    elif provider == "replicate":
        img_src = image_url if image_url else open(str(input_path), "rb")
        try:
            result = generate_video_replicate(
                model_config, img_src, prompt, duration, aspect_ratio,
                api_token=rep_token, use_obfuscation=use_obfuscation,
            )
        finally:
            if hasattr(img_src, "close"):
                img_src.close()

    video_url = None
    if result:
        if "video" in result and isinstance(result["video"], dict):
            video_url = result["video"].get("url")
        elif "video" in result and isinstance(result["video"], str):
            video_url = result["video"]
        elif "url" in result:
            video_url = result["url"]
    if not video_url:
        raise RuntimeError("No video URL in API response")
    local_path = save_video_from_url(video_url)
    if not local_path:
        raise RuntimeError("Failed to download video")
    return video_url, local_path


IMG2IMG_MODEL_ID = "fal-ai/fast-sdxl/image-to-image"  # kept for potential future use; no img2img runner here
