from telegram import Update, Poll
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    ContextTypes, ConversationHandler
)
import google.generativeai as genai
import re
import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import json
import logging

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
    "ADMIN_USER_ID": os.getenv("ADMIN_USER_ID"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "MODEL_NAME": "gemini-1.5-flash",
    "USER_DATA_FILE": "user_data.json",
    "PLANS": {
        "free": {
            "name": "Free",
            "daily_quizzes": 5,
            "max_questions": 10,
            "price": 0
        },
        "premium": {
            "name": "Premium",
            "daily_quizzes": float('inf'),
            "max_questions": 25,
            "price": 20  # EGP
        }
    },
    "DIFFICULTY_LEVELS": {
        "easy": "Generate simpler questions focusing on basic concepts",
        "medium": "Generate balanced questions with some challenging aspects",
        "hard": "Generate complex questions requiring deep understanding"
    }
}

# Conversation states
ASK_TEXT, ASK_NUM, ASK_DIFFICULTY = range(3)

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class UserManager:
    @staticmethod
    def load_user_data():
        try:
            with open(CONFIG["USER_DATA_FILE"], 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def save_user_data(user_data):
        with open(CONFIG["USER_DATA_FILE"], 'w') as f:
            json.dump(user_data, f, indent=2)

    @staticmethod
    def get_user(user_id, user_data):
        if str(user_id) not in user_data:
            user_data[str(user_id)] = {
                "subscription": "free",
                "quiz_count": 0,
                "last_quiz_date": None,
                "name": None
            }
        return user_data[str(user_id)]

    @staticmethod
    def reset_daily_counts(user_data):
        today = datetime.now().strftime("%Y-%m-%d")
        for user_id, data in user_data.items():
            if data["last_quiz_date"] != today:
                data["quiz_count"] = 0
                data["last_quiz_date"] = today
        return user_data

class QuizGenerator:
    @staticmethod
    async def generate_questions(text: str, num_questions: int, difficulty: str) -> list:
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
            response = model.generate_content(prompt)
            return QuizGenerator.parse_questions(response.text, num_questions)
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    @staticmethod
    def parse_questions(raw_response: str, num_questions: int) -> list:
        questions = []
        current_question = {}
        
        for line in raw_response.split('\n'):
            line = line.strip()
            
            if re.match(r'^\d+\.', line):
                if current_question:
                    questions.append(current_question)
                current_question = {
                    "question": re.sub(r'^\d+\.\s*', '', line),
                    "options": [],
                    "correct_option_id": None
                }
            elif re.match(r'^[A-D]\)', line):
                option_text = re.sub(r'^[A-D]\)\s*', '', line)
                current_question["options"].append(option_text)
            elif "ANSWER:" in line:
                match = re.search(r'ANSWER:\s*([A-D])', line, re.IGNORECASE)
                if match:
                    current_question["correct_option_id"] = ord(match.group(1).upper()) - ord('A')
        
        if current_question and len(current_question.get("options", [])) == 4:
            questions.append(current_question)
        
        return questions[:num_questions]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command with user status and options for international users."""
    user = update.effective_user
    user_data = UserManager.load_user_data()
    user_entry = UserManager.get_user(user.id, user_data)
    
    if user.first_name and not user_entry["name"]:
        user_entry["name"] = user.first_name
        UserManager.save_user_data(user_data)
    
    user_data = UserManager.reset_daily_counts(user_data)
    UserManager.save_user_data(user_data)
    
    plan = CONFIG["PLANS"][user_entry["subscription"]]
    remaining = plan["daily_quizzes"] - user_entry["quiz_count"]
    
    message = [
        f"👋 Welcome {user.first_name} to Note2Quiz!",
        f"📋 Your plan: {plan['name']}",
        f"📊 Today's usage: {user_entry['quiz_count']}/{plan['daily_quizzes']} quizzes",
        f"🔢 Max questions per quiz: {plan['max_questions']}",
    ]
    
    if user_entry["subscription"] == "free":
        message.extend([
            "",
            "💎 Upgrade to Premium and get:",
            f"- Unlimited daily quizzes (Free: {CONFIG['PLANS']['free']['daily_quizzes']}/day)",
            f"- Up to {CONFIG['PLANS']['premium']['max_questions']} questions per quiz",
            "",
            "Use /upgrade to learn more!",
        ])
    
    message.append("\n📚 Send me your lecture notes to generate a quiz!")
    
    await update.message.reply_text("\n".join(message))
    return ASK_TEXT
async def ask_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ask user to select difficulty level."""
    context.user_data['text'] = update.message.text
    
    keyboard = [
        ["Easy", "Medium", "Hard"]
    ]
    
    await update.message.reply_text(
        "📊 Choose difficulty level:",
        reply_markup={
            "keyboard": keyboard,
            "resize_keyboard": True,
            "one_time_keyboard": True
        }
    )
    return ASK_DIFFICULTY

async def ask_number(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ask for number of questions after difficulty is selected."""
    difficulty = update.message.text.lower()
    if difficulty not in CONFIG["DIFFICULTY_LEVELS"]:
        await update.message.reply_text("Please choose a valid difficulty: Easy, Medium, or Hard")
        return ASK_DIFFICULTY
    
    context.user_data['difficulty'] = difficulty
    
    user_data = UserManager.load_user_data()
    user_entry = UserManager.get_user(update.effective_user.id, user_data)
    plan = CONFIG["PLANS"][user_entry["subscription"]]
    
    await update.message.reply_text(
        f"✅ Difficulty set to {difficulty.capitalize()}!\n"
        f"How many questions should I generate? (1-{plan['max_questions']})\n\n"
        f"Quizzes remaining today: {plan['daily_quizzes'] - user_entry['quiz_count']}"
    )
    return ASK_NUM

async def generate_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send the quiz based on all inputs."""
    try:
        user_data = UserManager.load_user_data()
        user_entry = UserManager.get_user(update.effective_user.id, user_data)
        plan = CONFIG["PLANS"][user_entry["subscription"]]
        
        num_questions = int(update.message.text)
        
        if num_questions <= 0 or num_questions > plan["max_questions"]:
            await update.message.reply_text(
                f"❌ Please enter a number between 1 and {plan['max_questions']}"
            )
            return ASK_NUM

        text = context.user_data.get('text', '')
        difficulty = context.user_data.get('difficulty', 'medium')
        
        if not text:
            await update.message.reply_text("❌ Oops! I lost your notes. Please start over with /start")
            return ConversationHandler.END

        msg = await update.message.reply_text(f"🧠 Generating {num_questions} {difficulty} questions...")

        questions = await QuizGenerator.generate_questions(text, num_questions, difficulty)
        
        if not questions:
            await msg.edit_text("❌ Sorry, I couldn't generate questions. Please try different notes.")
            return ConversationHandler.END

        # Update user's quiz count
        user_entry["quiz_count"] += 1
        user_entry["last_quiz_date"] = datetime.now().strftime("%Y-%m-%d")
        UserManager.save_user_data(user_data)

        await msg.edit_text(f"✅ Generated {len(questions)} questions! Sending polls...")

        # Send questions with delay
        for i, q in enumerate(questions, 1):
            try:
                await context.bot.send_poll(
                    chat_id=update.effective_chat.id,
                    question=f"Q{i}: {q['question']}"[:300],
                    options=q["options"],
                    type=Poll.QUIZ,
                    correct_option_id=q["correct_option_id"],
                    is_anonymous=False,
                    explanation=f"Correct answer: {q['options'][q['correct_option_id']][:200]}"
                )
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error sending question {i}: {e}")

        remaining = plan["daily_quizzes"] - user_entry["quiz_count"]
        end_message = [
            "🎉 Quiz complete!",
            f"Quizzes remaining today: {remaining}",
            "",
            "Use /start to create another quiz."
        ]
        
        if user_entry["subscription"] == "free" and remaining <= 2:
            end_message.append(
                "\n⚠️ Low on free quizzes! Upgrade to Premium with /upgrade"
            )
            
        await update.message.reply_text("\n".join(end_message))
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("❌ Please enter a valid number.")
        return ASK_NUM

async def upgrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle upgrade command."""
    user_data = UserManager.load_user_data()
    user_entry = UserManager.get_user(update.effective_user.id, user_data)
    
    if user_entry["subscription"] == "premium":
        await update.message.reply_text("🌟 You're already a Premium user!")
    else:
        await update.message.reply_text(
            f"💎 Premium Plan: {CONFIG['PLANS']['premium']['price']} EGP/month\n\n"
            "Benefits:\n"
            f"- Unlimited daily quizzes\n"
            f"- Up to {CONFIG['PLANS']['premium']['max_questions']} questions per quiz\n"
            "- Priority support\n\n"
            "To upgrade: /pay"
        )

async def pay_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle payment instructions for international users."""
    await update.message.reply_text(
        "💳 To upgrade to the Premium Plan:\n\n"
        "Please contact our support at @N2845555 for payment instructions and assistance.\n\n"
        "You'll be upgraded within 1 hour after confirmation."
    )

async def admin_upgrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to upgrade users."""
    if str(update.effective_user.id) != CONFIG["ADMIN_USER_ID"]:
        await update.message.reply_text("❌ Admin only command.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /upgrade_user <user_id>")
        return
    
    user_id = context.args[0]
    user_data = UserManager.load_user_data()
    user_entry = UserManager.get_user(user_id, user_data)
    
    user_entry["subscription"] = "premium"
    UserManager.save_user_data(user_data)
    
    await update.message.reply_text(f"✅ User {user_id} upgraded to Premium!")
    
    try:
        await context.bot.send_message(
            chat_id=user_id,
            text="🎉 You've been upgraded to Premium!"
        )
    except Exception as e:
        logger.error(f"Couldn't notify user {user_id}: {e}")

def main():
    """Start the bot with organized handlers."""
    app = ApplicationBuilder().token(CONFIG["TOKEN"]).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_difficulty)],
            ASK_DIFFICULTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_number)],
            ASK_NUM: [MessageHandler(filters.TEXT & ~filters.COMMAND, generate_quiz)],
        },
        fallbacks=[]
    )

    # Command handlers
    command_handlers = [
        CommandHandler("upgrade", upgrade_command),
        CommandHandler("pay", pay_command),
        CommandHandler("upgrade_user", admin_upgrade),
    ]

    for handler in command_handlers:
        app.add_handler(handler)
    
    app.add_handler(conv_handler)

    logger.info("Note2Poll bot is running with improved organization! 🚀")
    app.run_polling()

if __name__ == '__main__':
    main()