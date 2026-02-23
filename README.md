# Image to Video 🎬

اپ ساده برای تبدیل تصویر به ویدیو با استفاده از **Kling 3** و **Sora 2** از طریق [fal.ai](https://fal.ai).

## قابلیت‌ها

- 📤 آپلود تصویر (JPG, PNG, WebP)
- 📝 پرامپت از پیش‌تعریف شده یا دلخواه
- ⏱️ مدت ویدیو: 5، 10، 15 ثانیه (Kling) یا 4، 8، 12 ثانیه (Sora)
- 🎭 ۶ پرامپت آماده با حرکات جذاب به سمت دوربین
- 🎬 چهار مدل: Kling 3 Standard، Kling 3 Pro، Sora 2، Seedance 1.5 Pro

## نصب

```bash
pip install -r requirements.txt
```

## راه‌اندازی

1. یک API Key از [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) بگیر
2. اپ را اجرا کن:

```bash
streamlit run app.py
```

3. API Key را در سایدبار وارد کن
4. تصویر آپلود کن، پرامپت انتخاب کن و «ساخت ویدیو» بزن

## قیمت (fal.ai)

- **Kling 3**: حدوداً $0.07–0.14 per second
- **Sora 2**: حدوداً $0.10 per second

جزئیات به‌روز را در [fal.ai](https://fal.ai) ببین.

---

## دپلوی با GitHub و Streamlit Community Cloud

1. **ریپو در GitHub بساز**
   - برو [github.com/new](https://github.com/new)
   - نام ریپو (مثلاً `image-to-video`) و Public را انتخاب کن و Create repository بزن.

2. **پوشه پروژه را با Git وصل کن و پوش بزن** (در ترمینال، داخل پوشه پروژه):

```powershell
cd D:\888imagetovideo
git init
git add .
git commit -m "Initial commit: Image to Video app"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

به‌جای `USERNAME` و `REPO_NAME` نام کاربری و نام ریپوی خودت را بگذار.

3. **دپلوی در Streamlit Cloud**
   - برو [share.streamlit.io](https://share.streamlit.io)
   - با GitHub لاگین کن و **Deploy an app** را بزن.
   - ریپو و برنچ `main` را انتخاب کن.
   - **Main file path** را `app.py` بگذار.
   - در **Advanced settings** → **Secrets** این را اضافه کن (کلید fal.ai را از [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) بگیر):

```
FAL_KEY = "KEY_ID:KEY_SECRET"
```

   - Deploy بزن. بعد از چند دقیقه اپ آنلاین می‌شود.

4. **تغییرات بعدی**: هر وقت کد را عوض کردی، کافی است `git add .` و `git commit -m "..."` و `git push` بزنی؛ Streamlit Cloud خودش دوباره دپلوی می‌کند.
