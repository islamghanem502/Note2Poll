from __future__ import annotations

"""
Note2Poll — Telegram bot (DigitalOcean ready) - النسخة المعدلة
- واجهة المستخدم بالعربية أو الإنجليزية (اختيار)
- الأسئلة المولدة تعتمد على لغة النص المدخل فقط
- JSON storage with atomic writes
- Async concurrency guard via asyncio.Lock
- Daily auto-reset of usage counters
- Local rotating backups + admin /backup_now
- Safer keyboards via ReplyKeyboardMarkup

Requirements (pip):
    python-telegram-bot==21.*
    google-generativeai
    python-dotenv

Environment (.env):
    TELEGRAM_BOT_TOKEN=...
    ADMIN_USER_ID=123456789
    GEMINI_API_KEY=...

Run:
    python bot.py
"""

import os
import re
import json
import shutil
import logging
from datetime import datetime
import asyncio
from typing import Dict, Any, List

from dotenv import load_dotenv
from telegram import Update, Poll, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

# ----------------------
# Setup & Configuration
# ----------------------
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
USER_DATA_FILE = os.path.join(DATA_DIR, os.getenv("USER_DATA_FILE", "user_data.json"))
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
LOG_DIR = os.getenv("LOG_DIR", "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("note2poll")

CONFIG: Dict[str, Any] = {
    "TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
    "ADMIN_USER_ID": str(os.getenv("ADMIN_USER_ID", "")),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-1.5-flash"),
    "PLANS": {
        "free": {"name": "Free", "name_ar": "مجاني", "daily_quizzes": 5, "max_questions": 10, "price": 0},
        "premium": {"name": "Premium", "name_ar": "المميز", "daily_quizzes": float("inf"), "max_questions": 25, "price": 20},
    },
    "DIFFICULTY_LEVELS": {
        "easy": "Generate simpler questions focusing on basic concepts",
        "medium": "Generate balanced questions with some challenging aspects",
        "hard": "Generate complex questions requiring deep understanding",
    },
    "DEMO_VIDEO_URL": "https://drive.google.com/file/d/1s9qQSdpJp0XtyGeRZfNFjm5OxpxNRsaW/view?usp=sharing",
}

if not CONFIG["TOKEN"]:
    raise SystemExit("TELEGRAM_BOT_TOKEN not set")
if not CONFIG["GEMINI_API_KEY"]:
    logger.warning("GEMINI_API_KEY not set — generation will fail")

# conversation states
SELECT_LANG, ASK_TEXT, ASK_NUM, ASK_DIFFICULTY = range(4)

# global file lock to serialize JSON writes
FILE_LOCK = asyncio.Lock()

# ----------------------
# Storage helpers (JSON)
# ----------------------
class Storage:
    path: str = USER_DATA_FILE

    @staticmethod
    def _safe_write_atomic(path: str, data: Dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)  # atomic on POSIX

    @classmethod
    async def load(cls) -> Dict[str, Any]:
        if not os.path.exists(cls.path):
            return {}
        try:
            with open(cls.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            return {}

    @classmethod
    async def save(cls, data: Dict[str, Any]) -> None:
        async with FILE_LOCK:
            try:
                cls._safe_write_atomic(cls.path, data)
            except Exception as e:
                logger.exception(f"Failed to save user data: {e}")
                raise

    @classmethod
    async def backup(cls) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(BACKUP_DIR, f"user_data-{ts}.json")
        async with FILE_LOCK:
            if os.path.exists(cls.path):
                shutil.copy2(cls.path, dest)
            else:
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        return dest

# ----------------------
# User manager
# ----------------------
class UserManager:
    @staticmethod
    def get_user(user_id: int | str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        uid = str(user_id)
        if uid not in user_data:
            user_data[uid] = {
                "subscription": "free",
                "quiz_count": 0,
                "last_quiz_date": None,
                "name": None,
                "lang": "en",  # default interface language
            }
        return user_data[uid]

    @staticmethod
    def reset_daily_counts(user_data: Dict[str, Any]) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        for uid, data in user_data.items():
            if data.get("last_quiz_date") != today:
                data["quiz_count"] = 0
                data["last_quiz_date"] = today
        return user_data

# ----------------------
# Quiz generator (Gemini)
# ----------------------
class QuizGenerator:
    @staticmethod
    async def generate(text: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
        model = genai.GenerativeModel(CONFIG["MODEL_NAME"])

        difficulty_prompt = CONFIG["DIFFICULTY_LEVELS"].get(difficulty, "")
        prompt = f"""
Generate exactly {num_questions} multiple choice quiz questions based on the following text.
{difficulty_prompt}

Rules:
1. Each question must be based directly on the provided text
2. Each question must have EXACTLY 4 options labeled A), B), C), D)
3. Mark the correct answer with "ANSWER: X)" (X is A, B, C, or D)
4. Format each question like this example:

1. What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid
ANSWER: B)

Text to base questions on:
{text}
"""
        try:
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", "") or ""
            return QuizGenerator.parse(raw, num_questions)
        except Exception as e:
            logger.exception(f"Gemini generation failed: {e}")
            return []

    @staticmethod
    def parse(raw: str, num_questions: int) -> List[Dict[str, Any]]:
        questions: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None

        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if re.match(r"^\d+\.\s*", s):
                if current and len(current.get("options", [])) == 4 and current.get("correct_option_id") is not None:
                    questions.append(current)
                current = {"question": re.sub(r"^\d+\.\s*", "", s), "options": [], "correct_option_id": None}
            elif re.match(r"^[A-D]\)\s*", s):
                if current is not None:
                    current["options"].append(re.sub(r"^[A-D]\)\s*", "", s))
            elif "ANSWER:" in s and current is not None:
                m = re.search(r"ANSWER:\s*([A-D])", s, flags=re.I)
                if m:
                    current["correct_option_id"] = ord(m.group(1).upper()) - ord('A')

        if current and len(current.get("options", [])) == 4 and current.get("correct_option_id") is not None:
            questions.append(current)

        return questions[:num_questions]

# ----------------------
# Bot handlers
# ----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_data = await Storage.load()
    entry = UserManager.get_user(user.id, user_data)

    # capture name once
    if user.first_name and not entry.get("name"):
        entry["name"] = user.first_name
        await Storage.save(user_data)

    # daily reset
    user_data = UserManager.reset_daily_counts(user_data)
    await Storage.save(user_data)

    # Ask for language if not set
    if not entry.get("lang"):
        kb = ReplyKeyboardMarkup([
            ["English 🇬🇧", "العربية 🇸🇦"],
        ], resize_keyboard=True, one_time_keyboard=True)
        
        await update.message.reply_text(
            "Please choose your language / الرجاء اختيار اللغة:",
            reply_markup=kb
        )
        return SELECT_LANG
    
    return await show_welcome_message(update, context, entry["lang"])

async def select_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_data = await Storage.load()
    entry = UserManager.get_user(user.id, user_data)
    
    lang_choice = update.message.text
    if "English" in lang_choice:
        entry["lang"] = "en"
    elif "العربية" in lang_choice:
        entry["lang"] = "ar"
    else:
        await update.message.reply_text("Please choose a valid language option")
        return SELECT_LANG
    
    await Storage.save(user_data)
    await update.message.reply_text(
        "Language set successfully! / تم تعيين اللغة بنجاح!",
        reply_markup=ReplyKeyboardRemove()
    )
    return await show_welcome_message(update, context, entry["lang"])

async def show_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str):
    user = update.effective_user
    user_data = await Storage.load()
    entry = UserManager.get_user(user.id, user_data)
    plan = CONFIG["PLANS"][entry["subscription"]]
    used = entry["quiz_count"]
    total = plan["daily_quizzes"] if plan["daily_quizzes"] != float("inf") else "∞"

    if lang == "ar":
        lines = [
            f"👋 أهلاً بك {user.first_name} في Note2Quiz!",
            f"📋 خطتك: {plan['name_ar']}",
            f"📊 الاستخدام اليومي: {used}/{total} اختبارات",
            f"🔢 الحد الأقصى للأسئلة: {plan['max_questions']}",
            "",
            "📚 أرسل لي ملاحظاتك أو محاضراتك وسأقوم بإنشاء اختبار لك!",
            "",
            "الأوامر المتاحة:",
            "/newQuiz - إنشاء اختبار جديد",
            "/demoVideo - مشاهدة فيديو تعليمي",
            "/language - تغيير لغة البوت",
            "/upgrade - الترقية إلى الخطة المميزة",
            "/pay - معلومات الدفع للترقية",
        ]

        if entry["subscription"] == "free":
            lines += [
                "",
                "💎 ترقية إلى المميز واحصل على:",
                f"- عدد غير محدود من الاختبارات اليومية (المجاني: {CONFIG['PLANS']['free']['daily_quizzes']}/يوم)",
                f"- حتى {CONFIG['PLANS']['premium']['max_questions']} سؤال لكل اختبار",
            ]
    else:
        lines = [
            f"👋 Welcome {user.first_name} to Note2Quiz!",
            f"📋 Your plan: {plan['name']}",
            f"📊 Today's usage: {used}/{total} quizzes",
            f"🔢 Max questions per quiz: {plan['max_questions']}",
            "",
            "📚 Send me your lecture notes to generate a quiz!",
            "",
            "Available commands:",
            "/newQuiz - Create a new quiz",
            "/demoVideo - Watch tutorial video",
            "/language - Change bot language",
            "/upgrade - Upgrade to Premium plan",
            "/pay - Payment information for upgrade",
        ]

        if entry["subscription"] == "free":
            lines += [
                "",
                "💎 Upgrade to Premium and get:",
                f"- Unlimited daily quizzes (Free: {CONFIG['PLANS']['free']['daily_quizzes']}/day)",
                f"- Up to {CONFIG['PLANS']['premium']['max_questions']} questions per quiz",
            ]

    await update.message.reply_text("\n".join(lines))
    return ConversationHandler.END

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = ReplyKeyboardMarkup([
        ["English 🇬🇧", "العربية 🇸🇦"],
    ], resize_keyboard=True, one_time_keyboard=True)
    
    await update.message.reply_text(
        "Please choose your language / الرجاء اختيار اللغة:",
        reply_markup=kb
    )
    return SELECT_LANG

async def demo_video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = await Storage.load()
    entry = UserManager.get_user(update.effective_user.id, user_data)
    lang = entry.get("lang", "en")
    
    video_url = CONFIG["DEMO_VIDEO_URL"]
    
    if lang == "ar":
        message = (
            "🎥 فيديو تعليمي: كيفية استخدام البوت\n\n"
            f"شاهد هذا الفيديو لمعرفة كيفية إنشاء اختبارات من ملاحظاتك:\n{video_url}\n\n"
            "بعد مشاهدة الفيديو، يمكنك البدء باستخدام /newQuiz"
        )
    else:
        message = (
            "🎥 Tutorial Video: How to use the bot\n\n"
            f"Watch this video to learn how to create quizzes from your notes:\n{video_url}\n\n"
            "After watching, you can start with /newQuiz"
        )
    
    await update.message.reply_text(message)

async def new_quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_data = await Storage.load()
    entry = UserManager.get_user(user.id, user_data)
    lang = entry.get("lang", "en")
    
    # Check daily limit
    plan = CONFIG["PLANS"][entry["subscription"]]
    if entry["quiz_count"] >= plan["daily_quizzes"] and plan["daily_quizzes"] != float("inf"):
        if lang == "ar":
            await update.message.reply_text("❌ لقد استنفذت عدد الاختبارات اليومية. جرب مرة أخرى غدًا أو قم بالترقية باستخدام /upgrade")
        else:
            await update.message.reply_text("❌ You've reached your daily quiz limit. Try again tomorrow or upgrade with /upgrade")
        return ConversationHandler.END
    
    if lang == "ar":
        await update.message.reply_text("📝 الرجاء إرسال النص أو الملاحظات التي تريد إنشاء اختبار منها:")
    else:
        await update.message.reply_text("📝 Please send the text or notes you want to create a quiz from:")
    
    return ASK_TEXT

async def ask_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = await Storage.load()
    entry = UserManager.get_user(update.effective_user.id, user_data)
    lang = entry.get("lang", "en")
    
    context.user_data['text'] = update.message.text

    if lang == "ar":
        kb = ReplyKeyboardMarkup([
            ["سهل", "متوسط", "صعب"],
        ], resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text("📊 اختر مستوى الصعوبة:", reply_markup=kb)
    else:
        kb = ReplyKeyboardMarkup([
            ["Easy", "Medium", "Hard"],
        ], resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text("📊 Choose difficulty level:", reply_markup=kb)
    
    return ASK_DIFFICULTY

async def ask_number(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = await Storage.load()
    entry = UserManager.get_user(update.effective_user.id, user_data)
    lang = entry.get("lang", "en")
    
    diff = update.message.text.lower()
    difficulty_map_ar = {"سهل": "easy", "متوسط": "medium", "صعب": "hard"}
    
    if lang == "ar":
        if diff in difficulty_map_ar:
            diff = difficulty_map_ar[diff]
        elif diff not in CONFIG["DIFFICULTY_LEVELS"]:
            await update.message.reply_text("الرجاء اختيار مستوى صعوبة صحيح: سهل، متوسط، أو صعب")
            return ASK_DIFFICULTY
    elif diff not in CONFIG["DIFFICULTY_LEVELS"]:
        await update.message.reply_text("Please choose a valid difficulty: Easy, Medium, or Hard")
        return ASK_DIFFICULTY

    context.user_data['difficulty'] = diff
    plan = CONFIG["PLANS"][entry["subscription"]]

    if lang == "ar":
        await update.message.reply_text(
            f"✅ تم تعيين الصعوبة إلى {diff.capitalize()}!\n"
            f"كم سؤال تريد أن أنشئ؟ (1-{plan['max_questions']})\n\n"
            f"الاختبارات المتبقية اليوم: "
            f"{(plan['daily_quizzes'] - entry['quiz_count']) if plan['daily_quizzes'] != float('inf') else '∞'}",
            reply_markup=ReplyKeyboardRemove(),
        )
    else:
        await update.message.reply_text(
            f"✅ Difficulty set to {diff.capitalize()}!\n"
            f"How many questions should I generate? (1-{plan['max_questions']})\n\n"
            f"Quizzes remaining today: "
            f"{(plan['daily_quizzes'] - entry['quiz_count']) if plan['daily_quizzes'] != float('inf') else '∞'}",
            reply_markup=ReplyKeyboardRemove(),
        )
    return ASK_NUM

async def generate_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = await Storage.load()
        entry = UserManager.get_user(update.effective_user.id, user_data)
        lang = entry.get("lang", "en")
        plan = CONFIG["PLANS"][entry["subscription"]]

        # validate number
        try:
            n = int(update.message.text)
        except ValueError:
            if lang == "ar":
                await update.message.reply_text("❌ الرجاء إدخال رقم صحيح.")
            else:
                await update.message.reply_text("❌ Please enter a valid number.")
            return ASK_NUM

        if n <= 0 or n > plan["max_questions"]:
            if lang == "ar":
                await update.message.reply_text(f"❌ الرجاء إدخال رقم بين 1 و {plan['max_questions']}")
            else:
                await update.message.reply_text(f"❌ Please enter a number between 1 and {plan['max_questions']}")
            return ASK_NUM

        text = context.user_data.get('text', '')
        diff = context.user_data.get('difficulty', 'medium')
        if not text:
            if lang == "ar":
                await update.message.reply_text("❌ لقد فقدت ملاحظاتك. الرجاء البدء من جديد باستخدام /newQuiz")
            else:
                await update.message.reply_text("❌ Oops! I lost your notes. Please start over with /newQuiz")
            return ConversationHandler.END

        if lang == "ar":
            msg = await update.message.reply_text(f"🧠 جاري إنشاء {n} أسئلة...")
        else:
            msg = await update.message.reply_text(f"🧠 Generating {n} questions...")

        questions = await QuizGenerator.generate(text, n, diff)
        if not questions:
            if lang == "ar":
                await msg.edit_text("❌ عذرًا، لم أتمكن من إنشاء الأسئلة. الرجاء تجربة ملاحظات مختلفة.")
            else:
                await msg.edit_text("❌ Sorry, I couldn't generate questions. Please try different notes.")
            return ConversationHandler.END

        # update usage
        entry["quiz_count"] += 1
        entry["last_quiz_date"] = datetime.now().strftime("%Y-%m-%d")
        await Storage.save(user_data)

        if lang == "ar":
            await msg.edit_text(f"✅ تم إنشاء {len(questions)} أسئلة! جاري إرسال الاختبارات...")
        else:
            await msg.edit_text(f"✅ Generated {len(questions)} questions! Sending polls...")

        for i, q in enumerate(questions, 1):
            try:
                # قص الخيارات لـ 100 حرف
                options = [opt[:100] for opt in q["options"]]

                # قص السؤال لـ 300 حرف
                question_text = f"Q{i}: {q['question']}"[:300]

                # قص الشرح لـ 200 حرف
                explanation_text = f"Correct answer: {options[q['correct_option_id']][:200]}"

                await context.bot.send_poll(
                    chat_id=update.effective_chat.id,
                    question=question_text,
                    options=options,
                    type=Poll.QUIZ,
                    correct_option_id=q["correct_option_id"],
                    is_anonymous=False,
                    explanation=explanation_text,
                )
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error sending question {i}: {e}")

        remaining = (
            (plan["daily_quizzes"] - entry["quiz_count"]) if plan["daily_quizzes"] != float("inf") else "∞"
        )
        
        if lang == "ar":
            end_lines = [
                "🎉 اكتمل الاختبار!",
                f"الاختبارات المتبقية اليوم: {remaining}",
                "",
                "استخدم /newQuiz لإنشاء اختبار آخر.",
            ]
            if entry["subscription"] == "free" and remaining != "∞" and remaining <= 2:
                end_lines.append("\n⚠️ لديك اختبارات مجانية قليلة! قم بالترقية إلى المميز باستخدام /upgrade")
        else:
            end_lines = [
                "🎉 Quiz complete!",
                f"Quizzes remaining today: {remaining}",
                "",
                "Use /newQuiz to create another quiz.",
            ]
            if entry["subscription"] == "free" and remaining != "∞" and remaining <= 2:
                end_lines.append("\n⚠️ Low on free quizzes! Upgrade to Premium with /upgrade")

        await update.message.reply_text("\n".join(end_lines))
        return ConversationHandler.END

    except Exception:
        logger.exception("generate_quiz crashed")
        if lang == "ar":
            await update.message.reply_text("⚠️ حدث خطأ غير متوقع. الرجاء المحاولة مرة أخرى.")
        else:
            await update.message.reply_text("⚠️ An unexpected error occurred. Please try again.")
        return ConversationHandler.END


# ----------------------
# Commands: upgrade/pay/admin
# ----------------------
async def upgrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = await Storage.load()
    entry = UserManager.get_user(update.effective_user.id, user_data)
    lang = entry.get("lang", "en")
    
    if entry["subscription"] == "premium":
        if lang == "ar":
            await update.message.reply_text("🌟 أنت بالفعل مشترك في الخطة المميزة!")
        else:
            await update.message.reply_text("🌟 You're already a Premium user!")
        return
    
    if lang == "ar":
        await update.message.reply_text(
            f"💎 الخطة المميزة:\n"
            "- 25 جنيه للشهر\n"
            "- 40 جنيه للشهرين\n"
            "- 60 جنيه لثلاثة أشهر\n\n"
            "المميزات:\n"
            f"- عدد غير محدود من الاختبارات اليومية\n"
            f"- حتى {CONFIG['PLANS']['premium']['max_questions']} سؤال لكل اختبار\n"
            "- دعم فوري\n\n"
            "📌 خارج مصر: يرجي التواصل لمعرفة الأسعار وطرق الدفع \n\n"
            "للترقية: /pay"
    )
    else:
        await update.message.reply_text(
            f"💎 Premium Plan:\n"
            "- 25 EGP for 1 month\n"
            "- 40 EGP for 2 months\n"
            "- 60 EGP for 3 months\n\n"
            "Benefits:\n"
            f"- Unlimited daily quizzes\n"
            f"- Up to {CONFIG['PLANS']['premium']['max_questions']} questions per quiz\n"
            "- Priority support\n\n"
            "📌 For non-Egyptians: Please contact us for pricing & payment methods\n\n"
            "To upgrade: /pay"
    )

async def pay_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = await Storage.load()
    entry = UserManager.get_user(update.effective_user.id, user_data)
    lang = entry.get("lang", "en")
    
    if lang == "ar":
        await update.message.reply_text(
            "💳 للترقية إلى الخطة المميزة:\n\n"
            "الرجاء التواصل مع الدعم على @Note2PollServiceBot لمعرفة طريقة الدفع والمساعدة.\n\n"
            "اترك رسالتك و سوف يتم الرد في اسرع وقت ممكن"
        )
    else:
        await update.message.reply_text(
            "💳 To upgrade to the Premium Plan:\n\n"
            "Please contact our support at @Note2PollServiceBot for payment instructions and assistance.\n\n"
            "Leave your message and we will reply as soon as possible"
        )

# --- admin helpers ---

def _admin_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_user.id) != CONFIG["ADMIN_USER_ID"]:
            user_data = await Storage.load()
            entry = UserManager.get_user(update.effective_user.id, user_data)
            lang = entry.get("lang", "en")
            if lang == "ar":
                await update.message.reply_text("❌ هذا الأمر مخصص للمشرف فقط.")
            else:
                await update.message.reply_text("❌ Admin only command.")
            return
        return await func(update, context)
    return wrapper

@_admin_only
async def admin_upgrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /upgrade_user <user_id>")
        return
    user_id = context.args[0]
    user_data = await Storage.load()
    entry = UserManager.get_user(user_id, user_data)
    entry["subscription"] = "premium"
    await Storage.save(user_data)
    await update.message.reply_text(f"✅ User {user_id} upgraded to Premium!")
    try:
        lang = entry.get("lang", "en")
        if lang == "ar":
            await context.bot.send_message(chat_id=user_id, text="🎉 تم ترقيتك إلى الخطة المميزة!")
        else:
            await context.bot.send_message(chat_id=user_id, text="🎉 You've been upgraded to Premium!")
    except Exception as e:
        logger.error(f"Couldn't notify user {user_id}: {e}")

@_admin_only
async def admin_downgrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /downgrade_user <user_id>")
        return
    user_id = context.args[0]
    user_data = await Storage.load()
    entry = UserManager.get_user(user_id, user_data)
    entry["subscription"] = "free"
    await Storage.save(user_data)
    await update.message.reply_text(f"✅ User {user_id} downgraded to Free!")
    try:
        lang = entry.get("lang", "en")
        if lang == "ar":
            await context.bot.send_message(
                chat_id=user_id, 
                text="⚠️ تم تغيير اشتراكك إلى الخطة المجانية."
        )
        else:
            await context.bot.send_message(
            chat_id=user_id, 
            text="⚠️ Your subscription has been changed to Free plan."
        )
    except Exception as e:
        logger.error(f"Couldn't notify user {user_id}: {e}")

@_admin_only
async def admin_backup_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = await Storage.backup()
    await update.message.reply_text(f"🗂️ Backup created: {os.path.basename(path)}")

# ----------------------
# App bootstrap
# ----------------------

def build_app():
    app = ApplicationBuilder().token(CONFIG["TOKEN"]).build()

    # Conversation handler for /start (language selection only)
    start_conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECT_LANG: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_language)],
        },
        fallbacks=[],
    )

    # Conversation handler for /language command
    language_conv = ConversationHandler(
        entry_points=[CommandHandler("language", language_command)],
        states={
            SELECT_LANG: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_language)],
        },
        fallbacks=[],
    )

    # Conversation handler for /newQuiz (quiz creation process)
    quiz_conv = ConversationHandler(
        entry_points=[CommandHandler("newQuiz", new_quiz_command)],
        states={
            ASK_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_difficulty)],
            ASK_DIFFICULTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_number)],
            ASK_NUM: [MessageHandler(filters.TEXT & ~filters.COMMAND, generate_quiz)],
        },
        fallbacks=[],
    )

    app.add_handler(start_conv)
    app.add_handler(language_conv)
    app.add_handler(quiz_conv)

    # public commands
    app.add_handler(CommandHandler("demoVideo", demo_video_command))
    app.add_handler(CommandHandler("upgrade", upgrade_command))
    app.add_handler(CommandHandler("pay", pay_command))

    # admin commands
    app.add_handler(CommandHandler("upgrade_user", admin_upgrade))
    app.add_handler(CommandHandler("downgrade_user", admin_downgrade))
    app.add_handler(CommandHandler("backup_now", admin_backup_now))

    logger.info("Note2Poll bot is running (DO-ready) 🚀")
    return app


def main():
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()