"""
Telegram bot for Image-to-Video: send photo → send prompt → receive video.
Uses same config (Fal/Replicate) as the Streamlit app. Queue + worker send result to user.
"""
import io
import os
import json
import threading
import shutil
from pathlib import Path
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    ContextTypes,
    filters,
)

from gen_core import (
    DATA_ROOT,
    VIDEO_DIR,
    load_config,
    MODELS,
    get_available_models,
    run_one_generation,
)

# ─── Bot storage (same machine as app) ───────────────────────────────────────
BOT_STATE_FILE = DATA_ROOT / "telegram_bot_state.json"
BOT_QUEUE_FILE = DATA_ROOT / "telegram_bot_queue.json"
BOT_CREDITS_FILE = DATA_ROOT / "telegram_bot_credits.json"
BOT_FREE_USERS_FILE = DATA_ROOT / "telegram_bot_free_users.json"

DEFAULT_CREDITS = 3
DURATION_BOT = 5
# Telegram: only Seedance 1.5 Pro; each video = 20 Stars
DEFAULT_MODEL_BOT = "Seedance 1.5 Pro"
ADMIN_USERNAME = "Poker_Radar"
FREE_USER_CREDITS = 999
STARS_PER_VIDEO = 20

# Telegram Stars: (payload, stars, credits)
STAR_PACKS = [
    ("credits_1", 20, 1),   # 1 video = 20 ⭐
    ("credits_2", 40, 2),   # 2 videos = 40 ⭐
    ("credits_5", 100, 5),  # 5 videos = 100 ⭐
]


def load_state() -> dict:
    if not BOT_STATE_FILE.exists():
        return {}
    try:
        return json.loads(BOT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(data: dict):
    BOT_STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_queue() -> list:
    if not BOT_QUEUE_FILE.exists():
        return []
    try:
        return json.loads(BOT_QUEUE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_queue(jobs: list):
    BOT_QUEUE_FILE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def get_user_state(user_id: int) -> dict:
    state = load_state()
    return state.get(str(user_id), {"state": "idle", "image_path": None})


def set_user_state(user_id: int, data: dict):
    state = load_state()
    state[str(user_id)] = {**get_user_state(user_id), **data}
    save_state(state)


# ─── Admin (Poker_Radar): no credit check, admin-only commands ─────────────────
def _admin_username() -> str:
    cfg = load_config()
    return (cfg.get("telegram_admin_username") or "").strip() or ADMIN_USERNAME


def is_admin(update: Update) -> bool:
    u = update.effective_user if update else None
    if not u:
        return False
    uname = (u.username or "").strip()
    return uname == _admin_username() or uname == ADMIN_USERNAME


# ─── Free users: no need to charge, unlimited use ─────────────────────────────
def load_free_users() -> list:
    if not BOT_FREE_USERS_FILE.exists():
        return []
    try:
        data = json.loads(BOT_FREE_USERS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_free_users(ids: list):
    BOT_FREE_USERS_FILE.write_text(json.dumps(ids, indent=2), encoding="utf-8")


def is_free_user(user_id: int, username: str | None = None) -> bool:
    free = load_free_users()
    sid = str(user_id)
    if sid in free:
        return True
    if username and (username.strip().lstrip("@") in free or username.strip() in free):
        return True
    return False


def add_free_user(uid_or_username: str) -> bool:
    uid_or_username = (uid_or_username or "").strip().lstrip("@")
    if not uid_or_username:
        return False
    free = load_free_users()
    if uid_or_username not in free:
        free.append(uid_or_username)
        save_free_users(free)
    return True


def remove_free_user(uid_or_username: str) -> bool:
    uid_or_username = (uid_or_username or "").strip().lstrip("@")
    free = load_free_users()
    if uid_or_username in free:
        free.remove(uid_or_username)
        save_free_users(free)
        return True
    return False


def get_user_credits(user_id: int, username: str | None = None) -> int:
    if is_free_user(user_id, username):
        return FREE_USER_CREDITS
    if not BOT_CREDITS_FILE.exists():
        return DEFAULT_CREDITS
    try:
        data = json.loads(BOT_CREDITS_FILE.read_text(encoding="utf-8"))
        return int(data.get(str(user_id), DEFAULT_CREDITS))
    except Exception:
        return DEFAULT_CREDITS


def deduct_credit(user_id: int, username: str | None = None) -> bool:
    if is_free_user(user_id, username):
        return True
    data = {}
    if BOT_CREDITS_FILE.exists():
        try:
            data = json.loads(BOT_CREDITS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    key = str(user_id)
    credits = int(data.get(key, DEFAULT_CREDITS))
    if credits <= 0:
        return False
    data[key] = credits - 1
    BOT_CREDITS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def add_credits_to_user(uid: str, amount: int) -> bool:
    if amount <= 0:
        return False
    data = {}
    if BOT_CREDITS_FILE.exists():
        try:
            data = json.loads(BOT_CREDITS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    key = str(uid).strip()
    data[key] = int(data.get(key, DEFAULT_CREDITS)) + amount
    BOT_CREDITS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def set_credits_user(uid: str, amount: int) -> bool:
    data = {}
    if BOT_CREDITS_FILE.exists():
        try:
            data = json.loads(BOT_CREDITS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[str(uid).strip()] = int(amount)
    BOT_CREDITS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def add_job(chat_id: int, user_id: int, image_path: str, prompt: str, model: str) -> str:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(user_id)
    jobs = load_queue()
    jobs.append({
        "id": job_id,
        "chat_id": chat_id,
        "user_id": user_id,
        "image_path": image_path,
        "prompt": prompt,
        "model": model,
        "duration": DURATION_BOT,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    })
    save_queue(jobs)
    return job_id


def set_job_done(job_id: str, output_path: str):
    jobs = load_queue()
    for j in jobs:
        if j.get("id") == job_id:
            j["status"] = "completed"
            j["video_path"] = output_path
            j["output_path"] = output_path
            break
    save_queue(jobs)


def get_job_by_id(job_id: str) -> dict | None:
    """Return a single job dict by id, or None if not found."""
    jobs = load_queue()
    for j in jobs:
        if j.get("id") == job_id:
            return j
    return None


def set_job_field(job_id: str, key: str, value):
    """Update one field of a job in the queue."""
    jobs = load_queue()
    for j in jobs:
        if j.get("id") == job_id:
            j[key] = value
            break
    save_queue(jobs)


def set_job_failed(job_id: str, error: str):
    jobs = load_queue()
    for j in jobs:
        if j.get("id") == job_id:
            j["status"] = "failed"
            j["error"] = error
            break
    save_queue(jobs)


def pop_next_pending_job() -> dict | None:
    jobs = load_queue()
    for i, j in enumerate(jobs):
        if j.get("status") == "pending":
            jobs[i]["status"] = "running"
            save_queue(jobs)
            return j
    return None


def _admin_chat_id() -> int | None:
    """Optional: forward errors to admin. Set telegram_admin_chat_id in config."""
    cfg = load_config()
    raw = cfg.get("telegram_admin_chat_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _send_error_to_admin_sync(app: Application, error_msg: str, context_str: str = ""):
    """Send error to admin chat if configured (from worker thread)."""
    admin = _admin_chat_id()
    if not admin:
        return
    try:
        msg = f"🔴 Bot Error\n{context_str}\n\n{error_msg[:800]}"
        # schedule on app event loop
        async def _send(bot, chat_id, text):
            try:
                await bot.send_message(chat_id=chat_id, text=text)
            except Exception:
                pass

        app.create_task(_send(app.bot, admin, msg))
    except Exception:
        pass


async def _send_video_async(
    bot, chat_id: int, local_path: str | None = None, video_url: str | None = None
):
    """Send video to user: try local file first (as bytes), then URL. On failure, send error message."""
    sent = False
    if local_path and Path(local_path).exists():
        try:
            data = Path(local_path).read_bytes()
            buf = io.BytesIO(data)
            buf.name = "video.mp4"
            await bot.send_video(
                chat_id=chat_id, video=buf, caption="ویدئو آماده است."
            )
            sent = True
        except Exception:
            pass
    if not sent and video_url:
        try:
            await bot.send_video(
                chat_id=chat_id, video=video_url, caption="ویدئو آماده است."
            )
            sent = True
        except Exception:
            pass
    if not sent:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="ویدئو ساخته شد ولی ارسال خودکار خطا داشت. لطفاً به پشتیبانی بگو یا دوباره درخواست بزن. /credits",
            )
        except Exception:
            pass


async def _send_text_async(bot, chat_id: int, text: str):
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        pass


def status_check_keyboard(job_id: str):
    """Inline button for user to check their video job status."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 بررسی وضعیت", callback_data=f"status_{job_id}")],
    ])


async def callback_status_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle 'بررسی وضعیت' button: show status, send video if done, 1-min cooldown."""
    query = update.callback_query
    data = (query.data or "").strip()
    if not data.startswith("status_"):
        await query.answer()
        return
    job_id = data[7:].strip()
    if not job_id:
        await query.answer("شناسه درخواست نامعتبر است.")
        return

    job = get_job_by_id(job_id)
    chat_id = query.message.chat.id if query.message else 0
    if not job:
        await query.answer("این درخواست دیگر وجود ندارد.")
        try:
            await query.edit_message_text("این درخواست دیگر در صف نیست.")
        except Exception:
            pass
        return
    if job.get("chat_id") != chat_id:
        await query.answer("این دکمه مربوط به این چت نیست.")
        return

    # 1-minute cooldown
    last_check = job.get("last_status_check")
    if last_check:
        try:
            t = datetime.fromisoformat(last_check)
            if (datetime.now() - t).total_seconds() < 60:
                await query.answer("لطفاً تا ۱ دقیقه دیگر دوباره بزن.", show_alert=True)
                return
        except Exception:
            pass
    set_job_field(job_id, "last_status_check", datetime.now().isoformat())

    status = job.get("status")
    # Still pending or running
    if status in ("pending", "running", None):
        created = job.get("created_at")
        msg = "ویدئوی شما هنوز در حال ساخت است. لطفاً چند دقیقه دیگر دوباره بررسی کن."
        if created:
            try:
                started = datetime.fromisoformat(created)
                mins = int((datetime.now() - started).total_seconds() // 60)
                if mins > 0:
                    msg = f"ویدئوی شما هنوز در حال ساخت است (~{mins} دقیقه گذشته). لطفاً صبر کن و بعد دوباره بزن."
            except Exception:
                pass
        await query.answer()
        try:
            await query.edit_message_text(msg, reply_markup=status_check_keyboard(job_id))
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=status_check_keyboard(job_id))
        return

    # Completed → send video and update message
    if status == "completed":
        await query.answer("در حال ارسال ویدئو…")
        video_path = job.get("video_path") or job.get("output_path")
        await _send_video_async(context.bot, chat_id, local_path=video_path)
        try:
            await query.edit_message_text("ویدئو آماده شد ✅")
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text="ویدئو آماده شد ✅")
        return

    # Failed
    if status == "failed":
        err = (job.get("error") or "")[:350]
        text = (
            "ساخت ویدئو ناموفق بود.\n\n"
            "دلیل: " + err + "\n\n"
            "پرامپت یا تصویر را عوض کن. برای شارژ /credits"
        )
        await query.answer()
        try:
            await query.edit_message_text(text)
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=text)
        return

# ─── Worker (runs in background thread) ───────────────────────────────────────
def worker_loop(app: Application):
    while True:
        job = pop_next_pending_job()
        if not job:
            import time
            time.sleep(5)
            continue
        job_id = job.get("id")
        chat_id = job.get("chat_id")
        image_path = job.get("image_path")
        prompt = job.get("prompt", "").strip()
        model = job.get("model", DEFAULT_MODEL_BOT)
        duration = job.get("duration", DURATION_BOT)
        try:
            video_url, local_path = run_one_generation(
                image_path, prompt, model, duration=duration, config=load_config()
            )
            # Mark job done and send video to user automatically.
            set_job_done(job_id, local_path)
            app.create_task(
                _send_video_async(
                    app.bot,
                    chat_id,
                    local_path=local_path,
                    video_url=video_url,
                )
            )
            # Remember that we already scheduled sending this video
            set_job_field(job_id, "sent", True)
        except Exception as e:
            err = str(e)
            set_job_failed(job_id, err)
            _send_error_to_admin_sync(app, err, f"job_id={job_id} chat_id={chat_id}")


# ─── Admin menu (inline keyboard) ──────────────────────────────────────────────
def admin_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ کاربر رایگان", callback_data="admin_addfree")],
        [InlineKeyboardButton("➖ حذف رایگان", callback_data="admin_removefree")],
        [InlineKeyboardButton("📋 لیست رایگان", callback_data="admin_listfree")],
        [InlineKeyboardButton("💰 اضافه اعتبار", callback_data="admin_addcredits")],
        [InlineKeyboardButton("👤 اعتبار کاربر", callback_data="admin_user")],
        [InlineKeyboardButton("📖 راهنما", callback_data="admin_help")],
    ])


async def callback_admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not update.effective_user or (update.effective_user.username or "").strip() != _admin_username():
        return
    data = (query.data or "").strip()
    if data == "admin_addfree":
        await query.edit_message_text("برای افزودن کاربر رایگان:\n/addfree <user_id یا @username>")
    elif data == "admin_removefree":
        await query.edit_message_text("برای حذف از رایگان:\n/removefree <user_id یا @username>")
    elif data == "admin_listfree":
        free = load_free_users()
        await query.edit_message_text("کاربران رایگان:\n" + ("\n".join(free) if free else "خالی"))
    elif data == "admin_addcredits":
        await query.edit_message_text("بعد از پرداخت کاربر:\n/addcredits <user_id> <تعداد>")
    elif data == "admin_user":
        await query.edit_message_text("بررسی اعتبار:\n/user <user_id>")
    elif data == "admin_help":
        await query.edit_message_text(
            "دستورات ادمین:\n"
            "/addfree, /removefree, /listfree\n"
            "/addcredits <id> <تعداد>, /setcredits <id> <تعداد>\n"
            "/user <id>\n\n/admin برای منو"
        )


# ─── Handlers ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_admin(update):
        await update.message.reply_text(
            "منوی ادمین — یکی از گزینه‌ها را انتخاب کن:",
            reply_markup=admin_menu_keyboard(),
        )
        return
    await update.message.reply_text(
        "سلام. یک *عکس* بفرست، بعد *پرامپت* (حرکت را توصیف کن). ویدئو اینجا می‌آید.\n\n/help | /credits | /pay",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "• یک *عکس* بفرست.\n"
        "• بعد *پرامپت* بفرست (حرکت یا صحنه را توصیف کن).\n"
        "• ویدئو با مدل Seedance 1.5 Pro ساخته می‌شود و اینجا می‌آید.\n\n"
        "هر ویدئو = ۲۰ ستاره. اعتبار: /credits | شارژ: /pay",
        parse_mode="Markdown",
    )


async def cmd_credits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    username = (update.effective_user.username or "") if update.effective_user else ""
    n = get_user_credits(user_id, username)
    if n <= 0:
        await update.message.reply_text(
            f"اعتبار شما: {n}.\n\nحساب شارژ نشده. برای شارژ با ادمین تماس بگیرید یا /pay"
        )
    else:
        await update.message.reply_text(f"اعتبار شما: {n} ویدئو. برای شارژ بیشتر: /pay")


def _pay_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("۱ ویدئو — 20 ⭐", callback_data="pay_credits_1")],
        [InlineKeyboardButton("۲ ویدئو — 40 ⭐", callback_data="pay_credits_2")],
        [InlineKeyboardButton("۵ ویدئو — 100 ⭐", callback_data="pay_credits_5")],
        [InlineKeyboardButton("تماس با ادمین", url=f"https://t.me/{_admin_username()}")],
    ])


async def cmd_pay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "شارژ با Telegram Stars (اتوماتیک) یا تماس با ادمین:",
        reply_markup=_pay_keyboard(),
    )


async def callback_pay_stars(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = (query.data or "").strip()
    if not data.startswith("pay_credits_"):
        return
    payload = data.replace("pay_", "", 1)
    pack = next((p for p in STAR_PACKS if p[0] == payload), None)
    if not pack:
        return
    payload_id, stars, credits = pack
    chat_id = query.message.chat.id if query.message else 0
    try:
        await context.bot.send_invoice(
            chat_id=chat_id,
            title="اعتبار ویدئو",
            description=f"{credits} ویدئو — پرداخت با Stars",
            payload=payload_id,
            provider_token="",
            currency="XTR",
            prices=[LabeledPrice(label=f"{credits} ویدئو", amount=stars)],
        )
    except Exception as e:
        await query.message.reply_text(f"خطا در ارسال فاکتور: {str(e)[:200]}")


async def handle_pre_checkout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.pre_checkout_query.answer(ok=True)


async def handle_successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sp = getattr(update.message, "successful_payment", None)
    if not sp:
        return
    payload = (sp.invoice_payload or "").strip()
    pack = next((p for p in STAR_PACKS if p[0] == payload), None)
    if not pack:
        await update.message.reply_text("پرداخت دریافت شد؛ لطفاً با پشتیبانی تماس بگیرید.")
        return
    _, __, credits = pack
    user_id = update.effective_user.id if update.effective_user else 0
    add_credits_to_user(str(user_id), credits)
    await update.message.reply_text(f"✅ شارژ انجام شد. {credits} ویدئو به حساب شما اضافه شد. /credits")


# Filter: only messages that contain successful_payment
class SuccessfulPaymentFilter(filters.MessageFilter):
    def filter(self, message):
        return bool(getattr(message, "successful_payment", None))


async def cmd_userid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    username = (update.effective_user.username or "") if update.effective_user else ""
    await update.message.reply_text(
        f"شناسه شما برای ادمین:\n\n`{user_id}`\n\n"
        + (f"یوزرنیم: @{username}\n\n" if username else "")
        + "ادمین با این شناسه می‌تواند شما را رایگان یا شارژ کند.",
        parse_mode="Markdown",
    )


# ─── Admin-only commands ─────────────────────────────────────────────────────
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text(
        "منوی ادمین — یکی از گزینه‌ها را انتخاب کن:",
        reply_markup=admin_menu_keyboard(),
    )


async def cmd_addfree(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    args = (context.args or [])
    if not args:
        await update.message.reply_text("استفاده: /addfree <user_id یا @username>")
        return
    target = " ".join(args).strip().lstrip("@")
    if add_free_user(target):
        await update.message.reply_text(f"✅ {target} به لیست رایگان اضافه شد.")
    else:
        await update.message.reply_text("خطا در اضافه کردن.")


async def cmd_removefree(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    args = (context.args or [])
    if not args:
        await update.message.reply_text("استفاده: /removefree <user_id یا @username>")
        return
    target = " ".join(args).strip().lstrip("@")
    if remove_free_user(target):
        await update.message.reply_text(f"✅ {target} از لیست رایگان حذف شد.")
    else:
        await update.message.reply_text(f"{target} در لیست رایگان نبود.")


async def cmd_listfree(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    free = load_free_users()
    if not free:
        await update.message.reply_text("لیست رایگان خالی است.")
        return
    await update.message.reply_text("کاربران رایگان:\n" + "\n".join(free))


async def cmd_addcredits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    args = (context.args or [])
    if len(args) < 2:
        await update.message.reply_text("استفاده: /addcredits <user_id> <تعداد>")
        return
    try:
        uid, amount = args[0].strip(), int(args[1])
        if amount <= 0:
            await update.message.reply_text("تعداد باید مثبت باشد.")
            return
        add_credits_to_user(uid, amount)
        await update.message.reply_text(f"✅ {amount} اعتبار به کاربر {uid} اضافه شد.")
    except ValueError:
        await update.message.reply_text("تعداد باید عدد باشد.")


async def cmd_setcredits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    args = (context.args or [])
    if len(args) < 2:
        await update.message.reply_text("استفاده: /setcredits <user_id> <تعداد>")
        return
    try:
        uid, amount = args[0].strip(), int(args[1])
        if amount < 0:
            await update.message.reply_text("تعداد نمی‌تواند منفی باشد.")
            return
        set_credits_user(uid, amount)
        await update.message.reply_text(f"✅ اعتبار کاربر {uid} روی {amount} تنظیم شد.")
    except ValueError:
        await update.message.reply_text("تعداد باید عدد باشد.")


async def cmd_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    args = (context.args or [])
    if not args:
        await update.message.reply_text("استفاده: /user <user_id>")
        return
    uid = args[0].strip()
    cred = get_user_credits(int(uid) if uid.isdigit() else 0, uid if not uid.isdigit() else None)
    free = is_free_user(int(uid) if uid.isdigit() else 0, uid if not uid.isdigit() else None)
    await update.message.reply_text(f"کاربر {uid}: اعتبار={cred}, رایگان={free}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        chat_id = update.effective_chat.id if update.effective_chat else 0
        photo = update.message.photo
        if not photo:
            await update.message.reply_text("یک عکس بفرست.")
            return
        username = (update.effective_user.username or "") if update.effective_user else ""
        credits = get_user_credits(user_id, username)
        largest = photo[-1]
        file = await context.bot.get_file(largest.file_id)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".jpg"
        path = VIDEO_DIR / f"tg_{user_id}_{ts}{ext}"
        await file.download_to_drive(path)
        set_user_state(user_id, {"state": "waiting_prompt", "image_path": str(path)})
        await update.message.reply_text(
            f"عکس دریافت شد. الان *پرامپت* بفرست (حرکت یا صحنه را توصیف کن).\n\n"
            f"اعتبار شما: {credits} ویدئو.",
            parse_mode="Markdown",
        )
    except Exception as e:
        err = str(e)
        await update.message.reply_text(f"خطا در دریافت عکس: {err[:300]}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id if update.effective_user else 0
        chat_id = update.effective_chat.id if update.effective_chat else 0
        text = (update.message.text or "").strip()
        if not text:
            return
        state = get_user_state(user_id)
        if state.get("state") != "waiting_prompt":
            await update.message.reply_text("اول یک *عکس* بفرست، بعد پرامپت.", parse_mode="Markdown")
            return
        image_path = state.get("image_path")
        if not image_path or not Path(image_path).exists():
            set_user_state(user_id, {"state": "idle", "image_path": None})
            await update.message.reply_text("عکس منقضی شد. دوباره عکس بفرست.")
            return
        username = (update.effective_user.username or "") if update.effective_user else ""
        if not is_admin(update):
            credits = get_user_credits(user_id, username)
            if credits <= 0:
                await update.message.reply_text(
                    "اعتبار شما تمام شده. برای شارژ یا اشتراک /credits"
                )
                return
            if not deduct_credit(user_id, username):
                await update.message.reply_text("اعتبار کافی نیست. /credits")
                return
        model = DEFAULT_MODEL_BOT
        set_user_state(user_id, {"state": "idle", "image_path": None})
        job_id = add_job(chat_id, user_id, image_path, text, model)
        await context.bot.send_message(
            chat_id=chat_id,
            text="در حال ساخت ویدئو… (مدل: Seedance 1.5 Pro، ۵ ثانیه). وقتی آماده شد با دکمهٔ زیر وضعیت را چک کن یا ویدئو را بگیر.",
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text="🔍 وضعیت ساخت ویدئو:",
            reply_markup=status_check_keyboard(job_id),
        )
    except Exception as e:
        err = str(e)
        await update.message.reply_text(f"خطا: {err[:350]}")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        cfg = load_config()
        token = (cfg.get("telegram_bot_token") or "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in env or telegram_bot_token in config.json")
        return
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    async def post_init(application: Application):
        from telegram import BotCommand
        await application.bot.set_my_commands([
            BotCommand("start", "شروع / منوی ادمین"),
            BotCommand("help", "راهنما"),
            BotCommand("credits", "اعتبار"),
            BotCommand("pay", "شارژ با Stars"),
            BotCommand("userid", "شناسه برای ادمین"),
        ])

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(MessageHandler(SuccessfulPaymentFilter(), handle_successful_payment))
    app.add_handler(PreCheckoutQueryHandler(handle_pre_checkout))
    app.add_handler(CallbackQueryHandler(callback_admin_menu, pattern="^admin_"))
    app.add_handler(CallbackQueryHandler(callback_status_check, pattern="^status_"))
    app.add_handler(CallbackQueryHandler(callback_pay_stars, pattern="^pay_credits_"))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("credits", cmd_credits))
    app.add_handler(CommandHandler("pay", cmd_pay))
    app.add_handler(CommandHandler("userid", cmd_userid))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("addfree", cmd_addfree))
    app.add_handler(CommandHandler("removefree", cmd_removefree))
    app.add_handler(CommandHandler("listfree", cmd_listfree))
    app.add_handler(CommandHandler("addcredits", cmd_addcredits))
    app.add_handler(CommandHandler("setcredits", cmd_setcredits))
    app.add_handler(CommandHandler("user", cmd_user))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # Start worker thread (processes queue and sends video to users)
    def run_worker():
        import time
        time.sleep(2)
        worker_loop(app)
    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    print("Bot running. Queue worker started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
