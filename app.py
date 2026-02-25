"""
Image to Video — Dark Mode
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cropper import st_cropper
import fal_client
import replicate
import tempfile
import os
import io
import html
import json
import base64
import requests
import threading
import shutil
from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime
import time

def _get_data_root() -> Path:
    """Use app dir if writable, else /tmp (e.g. Streamlit Community Cloud read-only fs)."""
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


_DATA_ROOT = _get_data_root()
VIDEO_DIR = _DATA_ROOT / "generated_videos"
HISTORY_FILE = _DATA_ROOT / "history.json"
PROMPTS_FILE = _DATA_ROOT / "prompts.json"
CONFIG_FILE = _DATA_ROOT / "config.json"
NOTIFICATIONS_FILE = _DATA_ROOT / "notifications.json"
JOB_STATUS_FILE = _DATA_ROOT / "job_status.json"
DRAFT_FILE = _DATA_ROOT / "draft.json"
DRAFT_IMAGE_PATH = VIDEO_DIR / "_draft_input.jpg"

HOMOGLYPH = str.maketrans({
    "a": "ɑ", "e": "е", "i": "і", "o": "о", "s": "ѕ", "c": "с", "p": "р",
    "x": "х", "y": "у", "A": "А", "E": "Е", "O": "О", "B": "В", "H": "Н",
})


def load_draft() -> dict:
    if not DRAFT_FILE.exists():
        return {}
    try:
        return json.loads(DRAFT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_draft(image_path: str | None = None, prompt_draft_new: str | None = None, preset_sel: str | None = None):
    data = load_draft()
    if image_path is not None:
        data["image_path"] = image_path if image_path else None
    if prompt_draft_new is not None:
        data["prompt_draft_new"] = prompt_draft_new
    if preset_sel is not None:
        data["preset_sel"] = preset_sel
    DRAFT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_job_status() -> dict:
    """Returns { 'jobs': [ {...}, ... ] }. Supports legacy single-job format."""
    if not JOB_STATUS_FILE.exists():
        return {"jobs": []}
    try:
        data = json.loads(JOB_STATUS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "jobs" in data:
            return data
        if isinstance(data, dict) and data.get("running") is not None:
            job = dict(data)
            job.setdefault("id", job.get("started_at", "1"))
            return {"jobs": [job]}
        return {"jobs": []}
    except Exception:
        return {"jobs": []}


def save_job_status(data: dict):
    JOB_STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_jobs_list() -> list:
    return load_job_status().get("jobs", [])


def get_running_jobs() -> list:
    return [j for j in get_jobs_list() if j.get("running")]


def add_job(job_data: dict):
    job_data = dict(job_data)
    job_data.setdefault("id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    data = load_job_status()
    data.setdefault("jobs", [])
    data["jobs"].append(job_data)
    save_job_status(data)
    return job_data["id"]


def update_job_progress(job_id: str, progress_label: str):
    data = load_job_status()
    for j in data.get("jobs", []):
        if j.get("id") == job_id:
            j["progress_label"] = progress_label
            break
    save_job_status(data)


def remove_job(job_id: str):
    data = load_job_status()
    data["jobs"] = [j for j in data.get("jobs", []) if j.get("id") != job_id]
    save_job_status(data)


def set_job_error(job_id: str, error: str):
    data = load_job_status()
    for j in data.get("jobs", []):
        if j.get("id") == job_id:
            j["running"] = False
            j["error"] = error
            break
    save_job_status(data)


def clear_job_status():
    if JOB_STATUS_FILE.exists():
        JOB_STATUS_FILE.unlink()


def clear_job_by_id(job_id: str):
    remove_job(job_id)


def run_generation_worker(job_data: dict):
    """Runs in background: upload image, call provider API, save to history. Survives browser close."""
    job_id = job_data.get("id", "")
    input_path = Path(job_data.get("input_image_path", ""))
    if not input_path.exists():
        set_job_error(job_id, "Input image file not found.")
        add_notification("Background job failed: image not found.", "error")
        return
    config = load_config()
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
    model_name = job_data.get("model", "")
    if model_name not in MODELS:
        set_job_error(job_id, f"Unknown model: {model_name}")
        return
    model_config = MODELS[model_name]
    provider = model_config.get("provider", "fal")
    prompt = job_data.get("prompt", "").strip()
    duration = int(job_data.get("duration", 5))
    aspect_ratio = job_data.get("aspect_ratio", "16:9")
    use_obfuscation = job_data.get("use_obfuscation", True)

    image_url = None
    if provider == "fal":
        if not api_key:
            set_job_error(job_id, "Fal API key not set in Settings.")
            add_notification("Background job failed: no API key.", "error")
            return
        try:
            image_url = fal_client.upload_file(str(input_path))
            if not image_url:
                set_job_error(job_id, "Failed to upload image.")
                add_notification("Background job failed: upload failed.", "error")
                return
        except Exception as e:
            set_job_error(job_id, f"Upload failed: {e}")
            add_notification(f"Background job failed: {e}", "error")
            return
    elif provider == "replicate":
        if not rep_token:
            set_job_error(job_id, "Replicate API token not set in Settings.")
            return
        if api_key:
            try:
                image_url = fal_client.upload_file(str(input_path))
            except Exception:
                image_url = None
        else:
            image_url = None
    def on_progress(status):
        _, label = _status_to_progress(status)
        try:
            update_job_progress(job_id, label)
        except Exception:
            pass

    try:
        result = None
        if provider == "fal":
            result = generate_video(
                model_config,
                image_url,
                prompt,
                duration,
                aspect_ratio,
                api_key=api_key,
                use_obfuscation=use_obfuscation,
                on_queue_update=on_progress,
            )
        elif provider == "replicate":
            try:
                update_job_progress(job_id, "Generating (Replicate)…")
            except Exception:
                pass
            img_src = image_url if image_url else open(str(input_path), "rb")
            try:
                result = generate_video_replicate(
                    model_config,
                    img_src,
                    prompt,
                    duration,
                    aspect_ratio,
                    api_token=rep_token,
                    use_obfuscation=use_obfuscation,
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
            set_job_error(job_id, "Unexpected API response (no video URL).")
            add_notification("Background job: no video in response.", "error")
            return
        local_path = save_video_from_url(video_url)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = input_path.suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        image_save_path = VIDEO_DIR / f"image_{ts}{ext}"
        try:
            shutil.copy2(input_path, image_save_path)
            image_path = str(image_save_path)
        except Exception:
            image_path = str(input_path)
        history = load_history()
        entry = {
            "video_url": video_url,
            "local_path": local_path,
            "image_path": image_path,
            "prompt": prompt,
            "model": model_name,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "timestamp": datetime.now().isoformat(),
        }
        history.insert(0, entry)
        save_history(history)
        remove_job(job_id)
        add_notification("Video generated successfully (background).", "success")
    except Exception as e:
        err_str = str(e).lower()
        if "content_policy_violation" in err_str or "content checker" in err_str or "flagged" in err_str:
            msg = "Prompt was flagged by content policy. Enable Obfuscate prompt in Settings or rephrase the prompt."
        else:
            msg = str(e)
        set_job_error(job_id, msg)
        add_notification(f"Background job failed: {msg}", "error")


def load_notifications(limit: int = 20) -> list:
    if NOTIFICATIONS_FILE.exists():
        try:
            data = json.loads(NOTIFICATIONS_FILE.read_text(encoding="utf-8"))
            return (data if isinstance(data, list) else [])[-limit:]
        except Exception:
            pass
    return []


def add_notification(message: str, notif_type: str = "success"):
    data = load_notifications(limit=500)
    data.append({
        "message": message,
        "type": notif_type,
        "timestamp": datetime.now().isoformat(),
    })
    NOTIFICATIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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

DEFAULT_PROMPTS = {
    "Slow push-in": "Slow cinematic push-in, subject gracefully moving closer to camera, soft intimate lighting, mesmerizing eye contact",
    "Sultry approach": "Sultry gaze toward camera, smooth dolly forward movement, subject leans slightly toward viewer, warm golden hour glow",
    "Confident walk-in": "Confident smile approaching camera, gentle forward motion, shallow depth of field, glamorous studio lighting",
    "Elegant turn": "Elegant turn facing camera, fluid motion, subject walks slowly toward lens, cinematic bokeh",
    "Cinematic dolly": "Cinematic push-in shot, subject slowly moves toward camera with alluring presence, dramatic lighting",
    "Romantic approach": "Emotional approach to camera, intimate movement forward, soft focus transition, romantic atmosphere",
}


def load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        try:
            return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(DEFAULT_PROMPTS)


def save_prompts(prompts: dict):
    PROMPTS_FILE.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

DURATION_OPTIONS = [5, 10, 15]

ASPECT_OPTIONS = {
    "Auto": "auto",
    "1:1": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
}

MODELS = {
    # fal.ai — Seedance (enable_safety_checker: false)
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
    # Replicate — Wan 2.1 Uncensored
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
    """Return model names for which ALL required keys are set."""
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


def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(history: list):
    VIDEO_DIR.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_from_history(index: int):
    """Remove a single history entry by index (0-based)."""
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        save_history(history)


def clear_history():
    """Remove all entries from history."""
    save_history([])


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


def save_input_image(uploaded_file) -> str | None:
    """Save uploaded image to generated_videos folder; return path for history."""
    if not uploaded_file:
        return None
    try:
        VIDEO_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = Path(uploaded_file.name).suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        path = VIDEO_DIR / f"image_{ts}{ext}"
        path.write_bytes(uploaded_file.getvalue())
        return str(path)
    except Exception:
        return None


def get_aspect_from_image(uploaded_file) -> str:
    if not uploaded_file:
        return "16:9"
    try:
        img = ImageOps.exif_transpose(Image.open(uploaded_file))
        w, h = img.size
        r = w / h if h else 1
        if 0.9 < r < 1.1:
            return "1:1"
        return "16:9" if r > 1 else "9:16"
    except Exception:
        return "16:9"


def get_aspect_from_path(path) -> str:
    if not path or not Path(path).exists():
        return "16:9"
    try:
        img = ImageOps.exif_transpose(Image.open(path))
        w, h = img.size
        r = w / h if h else 1
        if 0.9 < r < 1.1:
            return "1:1"
        return "16:9" if r > 1 else "9:16"
    except Exception:
        return "16:9"


def obfuscate_prompt(text: str) -> str:
    return text.translate(HOMOGLYPH)


def enhance_prompt(user_text: str, api_key: str) -> str | None:
    """Use fal LLM to rewrite user text as a stronger image-to-video prompt. Returns enhanced text or None on error."""
    if not (user_text or "").strip() or not (api_key or "").strip():
        return None
    system = (
        "You are an AI video generation prompt expert. "
        "Your task is to enhance the user's prompt strictly based on what they asked for. "
        "Do NOT add unrelated elements, characters, or actions. Focus purely on improving the requested scene, "
        "adding relevant camera movement, lighting, and style details. Keep it concise (1-3 sentences). "
        "Output ONLY the enhanced prompt, no explanation or quotes."
    )
    try:
        client = fal_client.SyncClient(key=api_key.strip())
        out = client.subscribe(
            "fal-ai/any-llm",
            arguments={
                "prompt": (user_text or "").strip(),
                "system_prompt": system,
                "model": "google/gemini-2.5-flash-lite",
                "max_tokens": 300,
            },
        )
        if out is None:
            return None
        if isinstance(out, str):
            return out.strip() or None
        if isinstance(out, dict):
            # fal-ai/any-llm returns {"output": "..."}
            data = out.get("data") or out
            text = data.get("output") or data.get("text") or data.get("result")
            if isinstance(text, list) and len(text) and isinstance(text[0], dict):
                text = text[0].get("content") or text[0].get("text")
            return (text or "").strip() or None
        return None
    except Exception:
        return None


CROPPED_IMAGE_PATH = VIDEO_DIR / "_cropped_input.jpg"


def _img_to_base64(path: str, max_size: int = 360) -> str:
    """Read an image, resize for preview, return base64 JPEG string."""
    try:
        img = ImageOps.exif_transpose(Image.open(path)).copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            if img.mode == "P":
                img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""



def get_image_url(uploaded_file) -> str | None:
    if not uploaded_file:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return fal_client.upload_file(tmp_path)
    finally:
        os.unlink(tmp_path)


def btn_group(key: str, options: list, default=None, fmt=None) -> object:
    """Custom pill-button group using st.columns + session_state."""
    if key not in st.session_state:
        st.session_state[key] = default if default is not None else options[0]

    cols = st.columns(len(options))
    for col, opt in zip(cols, options):
        label = fmt(opt) if fmt else str(opt)
        is_active = st.session_state[key] == opt
        btn_class = "pill-btn active" if is_active else "pill-btn"
        # render a real st.button so Streamlit handles click
        with col:
            if st.button(label, key=f"_btn_{key}_{opt}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state[key] = opt
                st.rerun()

    return st.session_state[key]


def _status_to_progress(status) -> tuple[float, str]:
    """Map fal Status to (progress 0..1, label)."""
    t = type(status).__name__
    if t == "Queued":
        pos = getattr(status, "position", 0)
        return 0.15, f"Queued (position {pos})…"
    if t == "InProgress":
        return 0.55, "Generating video…"
    if t == "Completed":
        return 1.0, "Done."
    return 0.3, "Processing…"


def generate_video(
    model_config: dict,
    image_url: str,
    prompt: str,
    user_duration: int,
    aspect_ratio: str,
    api_key: str = "",
    use_obfuscation: bool = True,
    on_queue_update=None,
) -> dict:
    """Fal.ai image-to-video."""
    duration = model_config["duration_map"].get(user_duration, user_duration)
    img_param = model_config.get("image_param", "image_url")
    api_prompt = obfuscate_prompt(prompt) if use_obfuscation else prompt

    params = {"prompt": api_prompt, img_param: image_url}
    params["duration"] = str(duration) if "kling" in model_config["id"].lower() else duration
    params["aspect_ratio"] = aspect_ratio

    if "kling" in model_config["id"].lower():
        params["generate_audio"] = False
    elif "seedance" in model_config["id"].lower():
        params["generate_audio"] = False
        params["enable_safety_checker"] = False
    elif "sora" in model_config["id"].lower():
        params["prompt"] = api_prompt + ", silent, no audio, no sound"

    client = fal_client.SyncClient(key=api_key.strip() if api_key else None)
    return client.subscribe(
        model_config["id"],
        arguments=params,
        with_logs=True,
        on_queue_update=on_queue_update,
    )


def generate_video_replicate(
    model_config: dict,
    image_source,
    prompt: str,
    user_duration: int,
    aspect_ratio: str,
    api_token: str,
    use_obfuscation: bool = True,
) -> dict:
    """Replicate image-to-video. image_source: URL string or open file handle."""
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


# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image to Video",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
iframe[title="streamlit_autorefresh.st_autorefresh"] {
    height:0 !important; min-height:0 !important; border:0 !important; position:absolute !important;
}

/* ─── TOKENS ─── */
:root {
    --bg:       #08080D;
    --surface:  #111118;
    --surface2: #17171F;
    --border:   #1F1F2E;
    --border2:  #28283A;
    --accent:   #7C6FFF;
    --accent2:  #9F6FFF;
    --text:     #E8E8EE;
    --text2:    #8888A0;
    --text3:    #44445A;
    --success:  #30D158;
    --danger:   #FF453A;
    --r-sm:     8px;
    --r-md:     12px;
    --r-lg:     16px;
}

html, body, .stApp { font-family:'Inter',-apple-system,sans-serif; background:var(--bg) !important; color:var(--text) !important; }

/* ─── LAYOUT ─── */
.main .block-container {
    max-width: 920px !important;
    padding: 1.25rem 1.75rem 4rem !important;
    margin: 0 auto;
}
.main .block-container:has(.page-history) {
    max-width: 1360px !important;
    padding: 1rem 1.75rem 2rem !important;
}
/* ─── HEADER ─── */
.app-header { text-align:center; padding:0.5rem 0 1.25rem; }
.app-header h1 {
    font-size:1.6rem; font-weight:700; letter-spacing:-0.04em;
    background:linear-gradient(135deg,#FFF 0%,#999 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0 0 0.2rem; line-height:1.2;
}
.app-header p { font-size:0.78rem; color:var(--text3); margin:0; }

/* ─── SECTION LABEL ─── */
.section-label {
    font-size:0.62rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:var(--text3);
    margin:1.1rem 0 0.45rem;
}
.form-model-label { font-size:0.9rem; font-weight:500; color:var(--text); margin:0.25rem 0; }

/* ─── INPUTS ─── */
.stTextArea textarea {
    background:var(--surface2) !important; border:1px solid var(--border2) !important;
    border-radius:var(--r-md) !important; color:var(--text) !important;
    font-size:0.875rem !important; line-height:1.55 !important; resize:vertical !important;
    transition:border-color 0.15s !important;
}
.stTextArea textarea:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(124,111,255,0.12) !important; outline:none !important;
}
.stTextArea textarea::placeholder { color:var(--text3) !important; }

.stTextInput input {
    background:var(--surface2) !important; border:1px solid var(--border2) !important;
    border-radius:var(--r-md) !important; color:var(--text) !important;
    font-size:0.875rem !important; transition:border-color 0.15s !important;
}
.stTextInput input:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(124,111,255,0.12) !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background:var(--surface2) !important; border:1px solid var(--border2) !important;
    border-radius:var(--r-md) !important; font-size:0.875rem !important; color:var(--text) !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within { border-color:var(--accent) !important; }

/* ─── BUTTONS ─── */
/* Primary — only the "Generate Video" button */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%) !important;
    color:#fff !important; font-weight:600 !important; font-size:0.9rem !important;
    padding:0.6rem 1.5rem !important; border-radius:var(--r-md) !important; border:none !important;
    box-shadow:0 2px 12px rgba(124,111,255,0.3) !important;
    transition:opacity 0.15s, transform 0.15s !important;
    width:100% !important;
}
.stButton > button[kind="primary"]:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }
.stButton > button[kind="primary"]:active { transform:translateY(0) !important; }

/* Secondary */
.stButton > button[kind="secondary"] {
    background:var(--surface) !important; border:1px solid var(--border2) !important;
    border-radius:var(--r-sm) !important; color:var(--text2) !important;
    font-size:0.82rem !important; font-weight:500 !important;
    padding:0.42rem 0.75rem !important; transition:all 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color:var(--accent) !important; color:var(--text) !important;
    background:rgba(124,111,255,0.07) !important;
}

/* Download */
.stDownloadButton > button {
    background:var(--surface2) !important; border:1px solid var(--border2) !important;
    border-radius:var(--r-sm) !important; color:var(--text2) !important;
    font-size:0.82rem !important; font-weight:500 !important;
    padding:0.42rem 0.75rem !important; transition:all 0.15s !important;
}
.stDownloadButton > button:hover { border-color:var(--accent) !important; color:var(--text) !important; }

/* ─── IMAGE / UPLOAD CARD ─── */
.img-card {
    width: 100%; max-width: 400px;
    margin-left: auto; margin-right: auto;
    border-radius: 16px; overflow: hidden;
    background: var(--surface2);
    border: 1px solid var(--border);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.img-card:hover { border-color: var(--border2); box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
.img-card .img-card-title {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text3);
    padding: 0.5rem 1rem; border-bottom: 1px solid var(--border);
    background: var(--surface);
}
.img-card [data-testid="stFileUploader"] { width: 100% !important; background: transparent !important; border: none !important; }
.img-card [data-testid="stFileUploaderDropzone"] {
    width: calc(100% - 1.5rem) !important; min-width: auto !important;
    height: 300px !important; min-height: 300px !important;
    margin: 0.75rem !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
    background: var(--surface) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
    cursor: pointer !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.img-card [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(124,111,255,0.06) !important;
}
.img-card [data-testid="stFileUploaderDropzone"] p {
    font-size: 0.68rem !important; margin: 0 !important;
    color: var(--text2) !important; text-align: center !important;
}
.img-card [data-testid="stFileUploaderDropzone"] small {
    font-size: 0.58rem !important; margin: 0 !important;
    color: var(--text3) !important; text-align: center !important; opacity: 0.9;
}
.img-card [data-testid="stFileUploaderDropzone"] button {
    font-size: 0.68rem !important; padding: 0.4rem 0.9rem !important;
    margin: 0.2rem 0 0 !important;
    border-radius: 8px !important;
    border: 1px solid var(--accent) !important;
    background: transparent !important;
    color: var(--accent) !important;
    transition: all 0.2s !important;
}
.img-card [data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--accent) !important; color: #fff !important;
}
.img-card [data-testid="stFileUploaderDropzone"] svg {
    width: 32px !important; height: 32px !important; opacity: 0.6;
}
.img-card .img-preview-wrap { padding: 0.75rem; padding-top: 0; }
.img-card .img-preview-inner {
    border-radius: 12px; overflow: hidden; background: var(--surface);
    aspect-ratio: 1; max-height: 280px;
    display: flex; align-items: center; justify-content: center;
}
.img-card .img-preview-inner img {
    max-width: 100%; max-height: 100%; object-fit: contain; display: block;
}
.img-card .img-toolbar {
    display: flex; gap: 8px;
    padding: 0.5rem 0.75rem 0.75rem;
    border-top: 1px solid var(--border);
    background: var(--surface);
}
.img-card .img-toolbar .stButton { flex: 1; }
.img-card .img-toolbar .stButton > button {
    width: 100% !important; padding: 0.4rem 0.6rem !important;
    font-size: 0.7rem !important; border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important; color: var(--text2) !important;
    transition: all 0.15s !important;
}
.img-card .img-toolbar .stButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: rgba(124,111,255,0.08) !important;
}
.img-card .img-crop-wrap {
    padding: 0.65rem 1rem; border-top: 1px solid var(--border);
    background: var(--surface); font-size: 0.68rem; color: var(--text3);
}

.image-box-wrap {
    width: 280px;
    min-height: 280px;
    border-radius: var(--r-lg);
    border: 1.5px dashed var(--border2);
    background: var(--surface);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.2s, background 0.2s;
}
.image-box-wrap:hover {
    border-color: var(--accent);
    background: rgba(124,111,255,0.02);
}
.image-box-wrap.has-image {
    border-style: solid;
    border-color: var(--border);
}
.image-box-wrap.has-image:hover {
    border-color: var(--border);
    background: var(--surface);
}
.img-card .img-preview-wrap + [data-testid="stFileUploader"],
.img-card.has-preview [data-testid="stFileUploader"] { display: none !important; }

[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    max-width: 100% !important;
}
[data-testid="stFileUploaderDropzone"] {
    width: 100% !important;
    min-width: 100% !important;
    height: 276px !important;
    min-height: 276px !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] p {
    color: var(--text3) !important;
    font-size: 0.82rem !important;
    margin: 0 0 0.25rem 0 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text3) !important;
    font-size: 0.7rem !important;
    opacity: 0.85;
}
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0.78rem !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--border2) !important;
    background: var(--surface2) !important;
    color: var(--text2) !important;
    margin-top: 0.6rem !important;
    transition: all 0.15s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* When image is loaded: same box shows image inside */
.image-box-preview {
    width: 100%;
    height: 252px;
    min-height: 252px;
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.image-box-preview img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
}
.image-box-actions {
    display: flex;
    gap: 6px;
    padding: 8px 10px;
    border-top: 1px solid var(--border);
    background: var(--surface2);
}
.image-box-actions .stButton > button {
    flex: 1;
    padding: 0.35rem 0.5rem !important;
    font-size: 0.75rem !important;
    border-radius: 6px !important;
}

.crop-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 0.75rem;
    margin-top: 0.5rem;
}

@media (max-width: 768px) {
    .img-card { max-width: 100%; }
    .img-card [data-testid="stFileUploaderDropzone"] {
        height: 260px !important; min-height: 260px !important;
    }
    .img-card .img-preview-inner { max-height: 260px; }
    .image-box-wrap { width: 220px; min-height: 220px; }
    [data-testid="stFileUploaderDropzone"] {
        height: 216px !important; min-height: 216px !important;
    }
    .image-box-preview { height: 192px; min-height: 192px; }
}

/* ─── PILL BUTTONS (duration / aspect) ─── */
.pill-btn {
    flex:1; min-width:52px; padding:0.38rem 0.5rem;
    border-radius:8px; border:1px solid var(--border);
    background:var(--surface2); color:var(--text2);
    font-size:0.8rem; font-weight:500;
    font-family:'Inter',sans-serif; text-align:center;
    cursor:pointer; transition:all 0.15s; user-select:none;
}
.pill-btn:hover { border-color:var(--accent); color:var(--text); background:rgba(124,111,255,0.07); }
.pill-btn.active { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }

/* ─── SIDEBAR HIDDEN (menu moved to top nav) ─── */
[data-testid="stSidebar"] {
    width: 0 !important; min-width: 0 !important; padding: 0 !important; overflow: hidden !important;
    border: none !important;
}
[data-testid="stSidebar"] > div:first-child { width: 0 !important; min-width: 0 !important; padding: 0 !important; }
button[kind="header"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ─── TOP NAV (first row in main) ─── */
.top-nav-wrap { margin: -0.5rem 0 0.5rem 0; }
.main .block-container [data-testid="stHorizontalBlock"]:first-of-type {
    align-items: center; padding: 0.6rem 0 0.9rem; margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--border); background: var(--bg);
    position: sticky; top: 0; z-index: 999; backdrop-filter: blur(8px);
}
.top-nav-logo { font-size: 1rem; font-weight: 700; color: #fff; letter-spacing: -0.03em; white-space: nowrap; }
.job-status-bar {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
    padding: 0.6rem 0.9rem; margin-bottom: 0.75rem; font-size: 0.78rem;
}
.job-status-bar .label { color: var(--accent); font-weight: 600; }
.sidebar-job-item {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--r-sm); padding: 0.5rem 0.65rem;
    margin-bottom: 0.35rem; font-size: 0.72rem; color: var(--text2); line-height: 1.4;
}
.sidebar-job-item strong { color: var(--text); }
.sidebar-job-progress { height: 3px; background: var(--border2); border-radius: 3px; overflow: hidden; margin-top: 0.25rem; }
.sidebar-job-progress-bar {
    height: 100%; width: 40%;
    background: linear-gradient(90deg,var(--accent),var(--accent2));
    border-radius: 3px; animation: sbprog 1.4s ease-in-out infinite;
}
@keyframes sbprog { 0%,100%{transform:translateX(0)} 50%{transform:translateX(130%)} }

/* ─── ALERTS ─── */
[data-testid="stSuccess"] {
    background:rgba(48,209,88,0.07) !important; border:1px solid rgba(48,209,88,0.18) !important;
    border-radius:var(--r-sm) !important; color:var(--success) !important; font-size:0.82rem !important;
}
[data-testid="stError"] {
    background:rgba(255,69,58,0.07) !important; border:1px solid rgba(255,69,58,0.18) !important;
    border-radius:var(--r-sm) !important; font-size:0.82rem !important;
}
[data-testid="stWarning"] {
    background:rgba(255,159,10,0.07) !important; border:1px solid rgba(255,159,10,0.18) !important;
    border-radius:var(--r-sm) !important; font-size:0.82rem !important;
}
[data-testid="stInfo"] {
    background:rgba(124,111,255,0.06) !important; border:1px solid rgba(124,111,255,0.15) !important;
    border-radius:var(--r-sm) !important; font-size:0.82rem !important;
}

/* ─── VIDEO / IMAGE ─── */
.stVideo { border-radius:var(--r-md) !important; overflow:hidden !important; border:1px solid var(--border) !important; }
.stImage img { border-radius:var(--r-md); border:1px solid var(--border); }

/* ─── EXPANDER ─── */
.streamlit-expanderHeader {
    background:var(--surface2) !important; border-radius:var(--r-sm) !important;
    border:1px solid var(--border) !important; font-size:0.78rem !important;
    font-weight:500 !important; color:var(--text) !important; padding:0.45rem 0.7rem !important;
}
.streamlit-expanderHeader:hover { border-color:var(--accent) !important; }
.streamlit-expanderContent {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-top:none !important; border-radius:0 0 var(--r-sm) var(--r-sm) !important; padding:0.6rem !important;
}

/* ─── HISTORY GRID (network grid, items side by side) ─── */
.page-history { display: block; }
.history-status-label { font-size: 0.62rem; font-weight: 600; color: var(--accent); margin-bottom: 0.15rem; }
.main .block-container:has(.page-history) [data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
.main .block-container:has(.page-history) [data-testid="stHorizontalBlock"] [data-testid="column"] {
    flex: 1 1 0 !important; min-width: 0 !important;
}
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) {
    background: linear-gradient(160deg, #13131C 0%, #0F0F16 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 0.3rem 0.4rem !important;
    margin-bottom: 0.35rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container):hover {
    border-color: rgba(124,111,255,0.35) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
}
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) img { max-height: 60px !important; object-fit: contain !important; }
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) .stVideo { max-height: 60px !important; }
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) .stCaption p { font-size: 0.56rem !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) .stExpander { margin-top: 0.2rem !important; }
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) .stExpander summary { padding: 0.25rem 0 !important; font-size: 0.62rem !important; }
.main .block-container:has(.page-history) [data-testid="column"] > div:has(.history-item-container) button { padding: 0.2rem 0.35rem !important; font-size: 0.58rem !important; }
.hist-copy-btn {
    font-size: 0.6rem !important; padding: 0.2rem 0.45rem !important;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--surface2); color: var(--text2);
    cursor: pointer; margin-top: 0.15rem;
}
.hist-copy-btn:hover { border-color: var(--accent); color: var(--accent); }

/* ─── MISC ─── */
hr { border:none !important; height:1px !important; background:var(--border) !important; margin:0.75rem 0 !important; }
.stCaption p, small { font-size:0.7rem !important; color:var(--text3) !important; }
label[data-testid="stWidgetLabel"] p {
    font-size:0.62rem !important; font-weight:700 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
    color:var(--text3) !important; margin-bottom:0.35rem !important;
}
[data-testid="stProgress"] > div > div { background:var(--border2) !important; border-radius:4px !important; height:4px !important; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; border-radius:4px !important; }
[data-testid="stSpinner"] > div { border-top-color:var(--accent) !important; }
.stCheckbox label { font-size:0.8rem !important; color:var(--text2) !important; }
[data-testid="stSidebar"] .stCaption { font-size:0.67rem !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div { background:var(--surface) !important; border-color:var(--border) !important; }

/* ═══ RESPONSIVE — MOBILE ══════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .main .block-container { max-width:100% !important; padding:0.6rem 0.85rem 2rem !important; }
    .main .block-container:has(.page-history) { max-width:100% !important; padding:0.5rem 0.75rem 1.5rem !important; }
    .main .block-container:has(.page-history) [data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:0.5rem !important; }
    .main .block-container:has(.page-history) [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:0 !important; }
    .app-header { padding: 0.4rem 0 0.9rem; }
    .app-header h1 { font-size:1.2rem; }
    .app-header p { font-size:0.72rem; }
    .main .block-container [data-testid="stHorizontalBlock"]:first-of-type { padding: 0.5rem 0 0.75rem; gap: 0.4rem; }
    .top-nav-logo { font-size: 0.9rem; }
    .img-card { max-width: 100% !important; }
    .img-card [data-testid="stFileUploaderDropzone"] { height: 220px !important; min-height: 220px !important; }
    .img-card .img-preview-inner { max-height: 240px; }
    .pill-btn { min-height: 44px; font-size: 0.8rem; padding: 0.5rem 0.6rem; }
    .stButton > button { min-height: 44px !important; }
    [data-testid="stFileUploaderDropzone"] { height: 220px !important; min-height: 220px !important; }
    .image-box-wrap { width: 100%; max-width: 280px; min-height: 220px; }
    .image-box-preview { height: 200px; min-height: 200px; }
    .section-label { margin: 0.9rem 0 0.35rem; font-size: 0.6rem; }
    .job-status-bar { padding: 0.5rem 0.75rem; font-size: 0.75rem; }
}
@media (max-width: 480px) {
    .main .block-container [data-testid="stHorizontalBlock"]:first-of-type { flex-wrap: wrap; gap: 0.5rem; }
    .main .block-container [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-child { min-width: 100%; }
    .main .block-container [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] { flex: 1 1 auto; min-width: 0; }
}

/* ═══ RESPONSIVE — LARGE DESKTOP ═══════════════════════════════════════════════ */
@media (min-width: 1400px) {
    .main .block-container { max-width:700px !important; }
    .main .block-container:has(.page-history) { max-width:1500px !important; }
}
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>Image to Video</h1>
        <p>Upload an image · Write a prompt · Generate your video</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("show_done_notification"):
        st.session_state["show_done_notification"] = False
        st.success("✓ Video generated successfully.")

    # ── Sidebar: menu only ──────────────────────────────────────────────────────
    config = load_config()
    try:
        if hasattr(st, "secrets") and st.secrets:
            for sec_key, cfg_key in (
                ("FAL_KEY", "api_key"),
                ("REPLICATE_API_TOKEN", "replicate_api_token"),
            ):
                if st.secrets.get(sec_key):
                    config[cfg_key] = str(st.secrets[sec_key]).strip()
    except Exception:
        pass
    available_models = get_available_models(config)
    all_model_names = list(MODELS.keys())
    if not available_models:
        available_models = all_model_names
    default_model = config.get("model") or available_models[0]
    if default_model not in available_models:
        default_model = available_models[0]
    model_list = available_models

    if "sidebar_page" not in st.session_state:
        st.session_state["sidebar_page"] = "generate"
    page = st.session_state["sidebar_page"]

    def go_generate():
        st.session_state["sidebar_page"] = "generate"

    def go_history():
        st.session_state["sidebar_page"] = "history"

    def go_settings():
        st.session_state["sidebar_page"] = "settings"

    def go_notifications():
        st.session_state["sidebar_page"] = "notifications"

    # ── Top nav (menu) ─────────────────────────────────────────────────────────
    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
    nc0, nc1, nc2, nc3, nc4 = st.columns([0.9, 1, 1, 1, 1])
    with nc0:
        st.markdown('<div class="top-nav-logo">✦ Image to Video</div>', unsafe_allow_html=True)
    with nc1:
        st.button(
            "Generate",
            type="primary" if page == "generate" else "secondary",
            key="menu_generate",
            use_container_width=True,
            on_click=go_generate,
        )
    with nc2:
        st.button(
            "History",
            type="primary" if page == "history" else "secondary",
            key="menu_history",
            use_container_width=True,
            on_click=go_history,
        )
    with nc3:
        st.button(
            "Settings",
            type="primary" if page == "settings" else "secondary",
            key="menu_settings",
            use_container_width=True,
            on_click=go_settings,
        )
    with nc4:
        st.button(
            "Notifications",
            type="primary" if page == "notifications" else "secondary",
            key="menu_notifications",
            use_container_width=True,
            on_click=go_notifications,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    kid = (config.get("key_id") or "").strip()
    ksec = (config.get("key_secret") or "").strip()
    raw_api = (config.get("api_key") or "").strip()
    if kid and ksec:
        api_key = f"{kid}:{ksec}"
    elif raw_api and ":" in raw_api:
        api_key = raw_api
    else:
        api_key = raw_api or ""
    model_name = default_model
    model_config = MODELS.get(model_name, list(MODELS.values())[0])
    if api_key:
        os.environ["FAL_KEY"] = api_key
    for env_name, cfg_name in (
        ("REPLICATE_API_TOKEN", "replicate_api_token"),
    ):
        v = (config.get(cfg_name) or "").strip()
        if v:
            os.environ[env_name] = v

    # Clear settings widget state when not on Settings so next open loads from config
    if page != "settings":
        for k in ("set_key_id", "set_key_secret", "set_model", "set_replicate_token"):
            st.session_state.pop(k, None)

    # ── Settings page ──────────────────────────────────────────────────────────
    if page == "settings":
        st.markdown('<div class="section-label">API Keys</div>', unsafe_allow_html=True)

        # Fal
        st.markdown('<div class="section-label" style="margin-top:0.5rem; font-size:0.85rem;">fal.ai (Kling, Sora, Seedance)</div>', unsafe_allow_html=True)
        if "set_key_id" not in st.session_state:
            st.session_state["set_key_id"] = kid or (raw_api.split(":", 1)[0] if ":" in raw_api else raw_api)
        if "set_key_secret" not in st.session_state:
            st.session_state["set_key_secret"] = ksec or (raw_api.split(":", 1)[-1] if ":" in raw_api else "")
        key_id = st.text_input("Key ID", type="password", placeholder="Key ID", label_visibility="collapsed", key="set_key_id")
        key_secret = st.text_input("Key Secret", type="password", placeholder="Key Secret", label_visibility="collapsed", key="set_key_secret")
        key_id = (key_id or "").strip()
        key_secret = (key_secret or "").strip()
        if key_id and key_secret:
            _api = f"{key_id}:{key_secret}"
        elif key_id and ":" in key_id:
            _api = key_id
        else:
            _api = key_id or ""
        st.caption("Key ID + Key Secret (or full key in Key ID).")
        st.markdown('<a href="https://fal.ai/dashboard/keys" target="_blank" style="font-size:0.75rem; color:#6C63FF;">fal.ai → Get API Key ↗</a>', unsafe_allow_html=True)

        # Replicate
        rep_token = (config.get("replicate_api_token") or "").strip()
        if "set_replicate_token" not in st.session_state:
            st.session_state["set_replicate_token"] = rep_token
        st.markdown('<div class="section-label" style="margin-top:0.75rem; font-size:0.85rem;">Replicate</div>', unsafe_allow_html=True)
        replicate_token = st.text_input("Replicate token", type="password", placeholder="r8_...", label_visibility="collapsed", key="set_replicate_token")
        replicate_token = (replicate_token or "").strip()
        st.caption("Token for Replicate image-to-video models (Wan, Minimax, etc.).")
        st.markdown('<a href="https://replicate.com/account/api-tokens" target="_blank" style="font-size:0.75rem; color:#6C63FF;">Replicate → API tokens ↗</a>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:0.75rem;">Default model</div>', unsafe_allow_html=True)
        set_model_index = model_list.index(model_name) if model_name in model_list else 0
        model_name_set = st.selectbox("Model", model_list, index=set_model_index, label_visibility="collapsed", key="set_model", format_func=lambda x: f"{MODELS.get(x, {}).get('badge', '')} {x}".strip())
        st.caption("All providers work independently. Fal key only needed for Seedance models.")
        if st.button("Save", type="primary", key="settings_save"):
            save_config(
                key_id=key_id,
                key_secret=key_secret,
                api_key=_api,
                model=model_name_set,
                replicate_api_token=replicate_token,
            )
            for k in ("set_key_id", "set_key_secret", "set_model", "set_replicate_token"):
                st.session_state.pop(k, None)
            st.success("Settings saved.")
            st.rerun()
        st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
        return

    # ── Notifications page ──────────────────────────────────────────────────────
    if st.session_state.get("sidebar_page") == "notifications":
        st.markdown('<div class="section-label">Notifications</div>', unsafe_allow_html=True)
        notifs = load_notifications(limit=50)
        if not notifs:
            st.info("No notifications yet.")
            st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
            return
        for n in reversed(notifs):
            msg = n.get("message", "")
            ts = n.get("timestamp", "")
            typ = n.get("type", "info")
            try:
                t = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            except Exception:
                t = (ts or "")[:16]
            icon = "✓" if typ == "success" else "✗" if typ == "error" else "•"
            label = f"{icon} {msg[:60]}{'…' if len(msg) > 60 else ''} — {t}"
            with st.expander(label, expanded=False):
                st.write(msg)
                st.caption(t)
        st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
        return

    # ── History page (4 per row, compact, modern) ───────────────────────────────
    if st.session_state.get("sidebar_page") == "history":
        st.markdown('<div class="page-history"></div>', unsafe_allow_html=True)
        col_title, col_ref, col_clear = st.columns([4, 1, 1])
        with col_title:
            st.markdown('<div class="section-label" style="margin-top:0;">History</div>', unsafe_allow_html=True)
        with col_ref:
            if st.button("↻ Refresh", use_container_width=True, key="hist_refresh"):
                st.rerun()
        with col_clear:
            if st.button("🗑 Clear history", use_container_width=True, key="hist_clear_all", help="Remove all videos from history"):
                clear_history()
                st.success("History cleared.")
                st.rerun()

        running = get_running_jobs()
        history = load_history()
        if not running and not history:
            st.info("No generated videos yet. Use **Generate** to create one.")
            st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
            return
        combined = [{"type": "running", "job": j} for j in running]
        for hi, h in enumerate(history):
            combined.append({"type": "completed", "h": h, "hist_idx": hi})
        n_cols = 4
        for row_start in range(0, len(combined), n_cols):
            row_items = combined[row_start : row_start + n_cols]
            cols = st.columns(n_cols)
            for c, item in enumerate(row_items):
                with cols[c]:
                    st.markdown('<div class="history-item-container">', unsafe_allow_html=True)
                    if item["type"] == "running":
                        j = item["job"]
                        inp_path = j.get("input_image_path") and Path(j.get("input_image_path", ""))
                        if inp_path and inp_path.exists():
                            st.image(str(inp_path), use_container_width=True)
                        st.markdown('<div class="history-status-label">⏳ ' + html.escape(j.get("progress_label") or "Connecting…") + '</div>', unsafe_allow_html=True)
                        st.progress(0.5, text="")
                        full_prompt = (j.get("prompt") or "").strip()
                        if full_prompt:
                            st.caption((full_prompt[:50] + "…" if len(full_prompt) > 50 else full_prompt))
                        meta = []
                        try:
                            meta.append(datetime.fromisoformat(j.get("started_at", "")).strftime("%H:%M"))
                        except Exception:
                            meta.append((j.get("started_at") or "")[:16])
                        if j.get("model"):
                            meta.append(j["model"])
                        if j.get("duration"):
                            meta.append(f"{j['duration']}s")
                        if meta:
                            st.caption(" · ".join(meta))
                    else:
                        h = item["h"]
                        i = item["hist_idx"]
                        img_path = h.get("image_path") and Path(h.get("image_path", ""))
                        has_img = img_path and img_path.exists()
                        video_src = h.get("local_path") if (h.get("local_path") and Path(h.get("local_path", "")).exists()) else h.get("video_url")
                        full_prompt = (h.get("prompt") or "").strip()
                        meta_parts = []
                        ts = h.get("timestamp", "")
                        if ts:
                            try:
                                meta_parts.append(datetime.fromisoformat(ts).strftime("%m/%d %H:%M"))
                            except Exception:
                                meta_parts.append(ts)
                        if h.get("model"):
                            meta_parts.append(h["model"])
                        if h.get("duration"):
                            meta_parts.append(f"{h['duration']}s")
                        if video_src:
                            st.video(video_src)
                        else:
                            st.caption("No video")
                        if full_prompt:
                            st.caption((full_prompt[:50] + "…" if len(full_prompt) > 50 else full_prompt))
                        if meta_parts:
                            st.caption(" · ".join(meta_parts))
                        if full_prompt:
                            prompt_b64 = base64.b64encode(full_prompt.encode("utf-8")).decode("ascii")
                            st.markdown(
                                f'<button type="button" class="hist-copy-btn" data-prompt-b64="{html.escape(prompt_b64)}" onclick="(function(){{var b=this;var t=atob(b.getAttribute(\'data-prompt-b64\'));navigator.clipboard.writeText(t).then(function(){{b.textContent=\'✓ Copied!\';setTimeout(function(){{b.textContent=\'📋 Copy prompt\';}},1200);}});}})();">📋 Copy prompt</button>',
                                unsafe_allow_html=True,
                            )
                        with st.expander("Prompt", expanded=False):
                            if full_prompt:
                                st.code(full_prompt, language=None)
                        data = None
                        local = h.get("local_path") and Path(h.get("local_path", "")).exists()
                        if local:
                            data = Path(h["local_path"]).read_bytes()
                        elif h.get("video_url"):
                            try:
                                data = requests.get(h["video_url"], timeout=30).content
                            except Exception:
                                pass
                        if data:
                            st.download_button("Download", data, file_name=f"video_{i}.mp4", mime="video/mp4", key=f"hist_dl_{i}", use_container_width=True)
                        if st.button("Use prompt", key=f"hist_use_prompt_{i}", use_container_width=True, help="Open Generate with this prompt"):
                            st.session_state["prefill_prompt"] = full_prompt
                            st.session_state["sidebar_page"] = "generate"
                            st.session_state["prompt_draft_new"] = full_prompt
                            st.session_state["show_prefill_message"] = True
                            st.rerun()
                        can_regen = has_img and img_path.exists()
                        if st.button("Regenerate", key=f"hist_regen_{i}", use_container_width=True, disabled=not can_regen, help="Same image + prompt" if can_regen else "Image not found"):
                            if can_regen:
                                job_data = {
                                    "running": True,
                                    "started_at": datetime.now().isoformat(),
                                    "prompt": full_prompt,
                                    "model": h.get("model", list(MODELS.keys())[0]),
                                    "duration": int(h.get("duration", 5)),
                                    "aspect_ratio": h.get("aspect_ratio", "16:9"),
                                    "use_obfuscation": True,
                                    "input_image_path": str(img_path),
                                }
                                job_id = add_job(job_data)
                                job_data["id"] = job_id
                                threading.Thread(target=run_generation_worker, args=(job_data,), daemon=True).start()
                                st.success("Regenerate started.")
                                st.rerun()
                        if st.button("🗑 Remove", key=f"hist_del_{i}", use_container_width=True, help="Remove this video from history"):
                            remove_from_history(i)
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
        return

    # ── Image-to-Image page removed ─────────────────────────────────────────────

    # ── Generate page (two-column: Image | Settings + Prompt) ───────────────────
    if st.session_state.pop("show_prefill_message", False):
        st.success("Prompt from History applied. Select an image and click Generate.")
    draft = load_draft()
    if draft and st.session_state.get("sidebar_page") == "generate":
        prev = st.session_state.get("_last_sidebar_page", "")
        if prev != "generate":
            if draft.get("prompt_draft_new"):
                st.session_state["prompt_draft_new"] = draft.get("prompt_draft_new", "")
            if draft.get("image_path") and Path(draft["image_path"]).exists():
                st.session_state["draft_image_path"] = draft["image_path"]

    col_img, col_form = st.columns([1, 1])

    uploaded = None
    with col_img:
        has_draft_img = bool(
            st.session_state.get("draft_image_path")
            and Path(st.session_state.get("draft_image_path", "")).exists()
        )
        crop_active = st.session_state.get("_crop_mode", False) if has_draft_img else False

        card_class = "img-card has-preview" if has_draft_img else "img-card"
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.markdown('<div class="img-card-title">Image</div>', unsafe_allow_html=True)

        if not has_draft_img:
            uploaded = st.file_uploader(
                "image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="img_uploader",
            )
            if uploaded:
                VIDEO_DIR.mkdir(exist_ok=True)
                try:
                    pil = ImageOps.exif_transpose(Image.open(uploaded))
                    if pil.mode in ("RGBA", "LA", "P"):
                        if pil.mode == "P":
                            pil = pil.convert("RGBA")
                        bg = Image.new("RGB", pil.size, (255, 255, 255))
                        bg.paste(pil, mask=pil.split()[-1] if pil.mode in ("RGBA", "LA") else None)
                        pil = bg
                    elif pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    pil.save(str(DRAFT_IMAGE_PATH), "JPEG", quality=95)
                except Exception:
                    DRAFT_IMAGE_PATH.write_bytes(uploaded.getvalue())
                st.session_state["draft_image_path"] = str(DRAFT_IMAGE_PATH)
                st.session_state.pop("_crop_mode", None)
                st.rerun()
        else:
            current_img_path = st.session_state["draft_image_path"]
            b64 = _img_to_base64(current_img_path)
            st.markdown(
                (
                    '<div class="img-preview-wrap">'
                    '<div class="img-preview-inner">'
                    f'<img src="data:image/jpeg;base64,{b64}" alt="" />'
                    "</div></div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown('<div class="img-toolbar">', unsafe_allow_html=True)
            ac1, ac2 = st.columns(2)
            with ac1:
                lbl = "✂ Close" if crop_active else "✂ Crop"
                if st.button(lbl, key="toggle_crop", use_container_width=True):
                    st.session_state["_crop_mode"] = not crop_active
                    st.rerun()
            with ac2:
                if st.button("🗑 Remove", key="clear_draft_img", use_container_width=True):
                    st.session_state.pop("draft_image_path", None)
                    st.session_state.pop("_crop_mode", None)
                    for p in [DRAFT_IMAGE_PATH, CROPPED_IMAGE_PATH]:
                        if p.exists():
                            p.unlink()
                    save_draft(image_path="")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            if crop_active:
                st.markdown('<div class="img-crop-wrap">', unsafe_allow_html=True)
                st.caption("Drag to select area, then click **Apply**.")
                try:
                    pil_img = ImageOps.exif_transpose(Image.open(current_img_path))
                    cropped = st_cropper(
                        pil_img,
                        box_color="#7C6FFF",
                        aspect_ratio=None,
                        key="img_cropper",
                    )
                    crop_btns = st.columns(2)
                    with crop_btns[0]:
                        if st.button("Apply Crop", type="primary", key="apply_crop", use_container_width=True):
                            if cropped is not None:
                                VIDEO_DIR.mkdir(exist_ok=True)
                                if cropped.mode in ("RGBA", "LA", "P"):
                                    if cropped.mode == "P":
                                        cropped = cropped.convert("RGBA")
                                    bg = Image.new("RGB", cropped.size, (255, 255, 255))
                                    bg.paste(cropped, mask=cropped.split()[-1] if cropped.mode in ("RGBA", "LA") else None)
                                    cropped = bg
                                elif cropped.mode != "RGB":
                                    cropped = cropped.convert("RGB")
                                cropped.save(str(CROPPED_IMAGE_PATH), "JPEG", quality=95)
                                st.session_state["draft_image_path"] = str(CROPPED_IMAGE_PATH)
                                st.session_state["_crop_mode"] = False
                                save_draft(image_path=str(CROPPED_IMAGE_PATH))
                                st.success("Image cropped.")
                                st.rerun()
                    with crop_btns[1]:
                        if st.button("Cancel", key="cancel_crop", use_container_width=True, help="Exit crop without applying"):
                            st.session_state["_crop_mode"] = False
                            st.rerun()
                except Exception as exc:
                    st.error(f"Crop error: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    draft_image_path = st.session_state.get("draft_image_path")

    with col_form:
        st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
        gen_model_index = model_list.index(model_name) if model_name in model_list else 0
        selected_model = st.selectbox(
            "Model",
            model_list,
            index=gen_model_index,
            label_visibility="collapsed",
            key="generate_model_sel",
            format_func=lambda x: f"{MODELS.get(x, {}).get('badge', '')} {x}".strip(),
        )
        if selected_model != model_name:
            config["model"] = selected_model
            save_config(
                key_id=(config.get("key_id") or ""),
                key_secret=(config.get("key_secret") or ""),
                api_key=(config.get("api_key") or ""),
                model=selected_model,
                replicate_api_token=(config.get("replicate_api_token") or ""),
            )
            model_name = selected_model
            model_config = MODELS.get(selected_model, list(MODELS.values())[0])
        form_model_config = MODELS.get(selected_model, model_config)
        st.markdown('<div class="section-label">Duration</div>', unsafe_allow_html=True)
        duration = btn_group("duration", DURATION_OPTIONS, default=5, fmt=lambda x: f"{x}s")
        st.markdown('<div class="section-label">Aspect ratio</div>', unsafe_allow_html=True)
        aspect_choice = btn_group("aspect", list(ASPECT_OPTIONS.keys()), default="Auto")
        aspect_raw = ASPECT_OPTIONS[aspect_choice]
        if aspect_raw == "auto":
            aspect_ratio = get_aspect_from_image(uploaded) if uploaded else (get_aspect_from_path(draft_image_path) if draft_image_path else "16:9")
        elif form_model_config.get("id") and "sora" in str(form_model_config["id"]).lower() and aspect_raw == "1:1":
            aspect_ratio = "9:16"
        else:
            aspect_ratio = aspect_raw

        prompts = load_prompts()
        
        st.markdown('<div class="section-label">Prompt</div>', unsafe_allow_html=True)
        
        if st.session_state.get("prefill_prompt") is not None:
            default_prompt = st.session_state.pop("prefill_prompt", "") or ""
            st.session_state["new_prompt_ta"] = default_prompt
        elif st.session_state.get("enhanced_prompt"):
            default_prompt = st.session_state.pop("enhanced_prompt", "") or ""
            st.session_state["new_prompt_ta"] = default_prompt
        else:
            default_prompt = st.session_state.get("prompt_draft_new", "")

        prompt = st.text_area("prompt", value=default_prompt, placeholder="Describe camera movement and scene...", label_visibility="collapsed", height=100, key="new_prompt_ta")
        st.session_state["prompt_draft_new"] = prompt

        col_enh, col_save = st.columns([1, 1])
        with col_enh:
            if st.button("✨ Enhance", key="btn_enhance_prompt", use_container_width=True):
                if prompt.strip() and api_key:
                    with st.spinner("Enhancing…"):
                        enhanced = enhance_prompt(prompt.strip(), api_key)
                    if enhanced:
                        st.session_state["enhanced_prompt"] = enhanced
                        st.success("Enhanced!")
                        st.rerun()
                    else:
                        st.error("Enhancement failed.")
        
        with col_save:
            with st.popover("💾 Save Preset", width="stretch"):
                new_name = st.text_input("Name", placeholder="Preset name...", label_visibility="collapsed")
                if st.button("Save", use_container_width=True, key="save_preset_btn"):
                    if new_name.strip() and prompt.strip():
                        prompts[new_name.strip()] = prompt.strip()
                        save_prompts(prompts)
                        st.success("Saved.")
                        st.rerun()

        with st.expander("📚 Load / Delete Presets"):
            if prompts:
                sel_preset = st.selectbox("Select Preset", list(prompts.keys()), label_visibility="collapsed")
                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    if st.button("Load", use_container_width=True, key="load_preset_btn"):
                        st.session_state["prefill_prompt"] = prompts[sel_preset]
                        st.rerun()
                with pc2:
                    if st.button("🗑", use_container_width=True, key="del_preset_btn"):
                        del prompts[sel_preset]
                        save_prompts(prompts)
                        st.success("Deleted.")
                        st.rerun()
            else:
                st.caption("No presets saved yet.")

        save_draft(image_path=draft_image_path or "", prompt_draft_new=st.session_state.get("prompt_draft_new", ""))

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        if st.button("Generate Video", type="primary", use_container_width=True):
            if not prompt or not str(prompt).strip():
                st.error("Please enter a prompt.")
                st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
                return
            has_image = bool(uploaded or (draft_image_path and Path(draft_image_path).exists()))
            if not has_image:
                st.error("Please upload an image.")
                st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
                return
            prov = form_model_config.get("provider", "fal")
            missing = None
            if prov == "fal" and not api_key:
                missing = "Fal API Key"
            elif prov == "replicate" and not (config.get("replicate_api_token") or "").strip():
                missing = "Replicate API Token"
            if missing:
                st.error(f"Please set {missing} in Settings.")
                st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")
                return
            VIDEO_DIR.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if uploaded:
                ext = Path(uploaded.name).suffix or ".jpg"
                if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"
                pending_path = VIDEO_DIR / f"_pending_{ts}{ext}"
                pending_path.write_bytes(uploaded.getvalue())
            else:
                ext = Path(draft_image_path).suffix or ".jpg"
                if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"
                pending_path = VIDEO_DIR / f"_pending_{ts}{ext}"
                shutil.copy2(draft_image_path, pending_path)
            job_data = {
                "running": True,
                "started_at": datetime.now().isoformat(),
                "prompt": str(prompt).strip(),
                "model": selected_model,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "use_obfuscation": True,
                "input_image_path": str(pending_path),
            }
            job_id = add_job(job_data)
            job_data["id"] = job_id
            threading.Thread(target=run_generation_worker, args=(job_data,), daemon=True).start()
            st.success("Video generation started. Check **History** when ready.")
            st.rerun()

    # Full width: latest video (if any)
    if "last_result" in st.session_state and st.session_state["last_result"]:
        r = st.session_state["last_result"]
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Latest Video</div>', unsafe_allow_html=True)
        video_src = r.get("local_path") if r.get("local_path") and Path(r["local_path"]).exists() else r["video_url"]
        st.video(video_src)
        local = r.get("local_path") and Path(r["local_path"]).exists()
        dl_data = Path(r["local_path"]).read_bytes() if local else requests.get(r["video_url"]).content
        st.download_button("Download Video", dl_data, file_name="video.mp4", mime="video/mp4", key="dl_current")

    # ── Under Generate: progress bar + notifications (errors) ──────────────────
    running = get_running_jobs()
    failed = [j for j in get_jobs_list() if not j.get("running") and j.get("error")]
    if running:
        st_autorefresh(interval=4000, limit=500, key="generate_autorefresh")
        st.markdown('<div class="job-status-bar"><span class="label">Generating</span></div>', unsafe_allow_html=True)
        for idx, j in enumerate(running):
            try:
                t = datetime.fromisoformat(j.get("started_at", "")).strftime("%H:%M")
            except Exception:
                t = (j.get("started_at") or "")[:16]
            label = html.escape((j.get("progress_label") or "Connecting…")[:40])
            model_short = html.escape((j.get("model") or "")[:20])
            st.markdown(
                f'<div class="sidebar-job-item">'
                f'<strong>#{idx+1}</strong> {model_short}<br/>'
                f'<span style="color:#8B7CF8">{label}</span> · <span style="opacity:0.5">{t}</span>'
                f'<div class="sidebar-job-progress"><div class="sidebar-job-progress-bar"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    if failed:
        st.markdown('<div class="job-status-bar"><span class="label">Status</span></div>', unsafe_allow_html=True)
        for j in failed:
            err = (j.get("error") or "")[:80] + ("…" if len(j.get("error", "")) > 80 else "")
            st.error("Error: " + err)
            if st.button("Dismiss", key=f"dismiss_job_{j.get('id', '')}"):
                clear_job_by_id(j.get("id", ""))
                st.success("Error status cleared.")
                st.rerun()

    # Notifications (recent) under Generate
    notifs = load_notifications(limit=8)
    if notifs:
        st.markdown('<div class="job-status-bar"><span class="label">Notifications</span></div>', unsafe_allow_html=True)
        for n in reversed(notifs):
            msg = n.get("message", "")
            ts = n.get("timestamp", "")
            typ = n.get("type", "info")
            try:
                t = datetime.fromisoformat(ts).strftime("%H:%M") if ts else ""
            except Exception:
                t = (ts or "")[:16]
            icon = "✓" if typ == "success" else "✗" if typ == "error" else "•"
            label = f"{icon} {msg[:50]}{'…' if len(msg) > 50 else ''} — {t}"
            with st.expander(label, expanded=False):
                st.write(msg)
                st.caption(ts)

    st.session_state["_last_sidebar_page"] = st.session_state.get("sidebar_page", "generate")


if __name__ == "__main__":
    main()
