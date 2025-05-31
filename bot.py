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

# Load API keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7592266138:AAEedzFl7SJeV_i4ETVlJ3Wai6pHMbPsuhc")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use the correct model name
MODEL_NAME = "gemini-1.5-flash"  # Updated to current model name

# Conversation states
ASK_TEXT, ASK_NUM = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the conversation and ask for lecture notes."""
    await update.message.reply_text(
        "📚 Welcome to Note2Poll! Send me your lecture notes/textbook content "
        "and I'll turn it into quiz questions!"
    )
    return ASK_TEXT

async def get_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Store the lecture text and ask for number of questions."""
    context.user_data['text'] = update.message.text
    await update.message.reply_text(
        "✅ Notes received! How many questions should I generate? (1-10)"
    )
    return ASK_NUM

async def generate_questions(text: str, num_questions: int) -> list:
    """Generate quiz questions from lecture text using Gemini with robust parsing."""
    # Initialize model
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""
Generate exactly {num_questions} multiple choice quiz questions based on the following text.
Follow these rules STRICTLY:

1. Each question must be based directly on the provided text
2. Each question must have EXACTLY 4 options labeled A), B), C), D)
3. Mark the correct answer with "ANSWER: X)" (where X is A, B, C, or D)
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
        raw_response = response.text
        print(f"Raw AI response:\n{raw_response}")  # For debugging
        
        # Parse the response
        questions = []
        current_question = {}
        
        for line in raw_response.split('\n'):
            line = line.strip()
            
            # Detect question start
            if re.match(r'^\d+\.', line):
                if current_question:
                    questions.append(current_question)
                current_question = {
                    "question": re.sub(r'^\d+\.\s*', '', line),
                    "options": [],
                    "correct_option_id": None
                }
            
            # Detect option (A), B), etc.)
            elif re.match(r'^[A-D]\)', line):
                option_text = re.sub(r'^[A-D]\)\s*', '', line)
                current_question["options"].append(option_text)
            
            # Detect correct answer
            elif "ANSWER:" in line:
                match = re.search(r'ANSWER:\s*([A-D])', line, re.IGNORECASE)
                if match:
                    letter = match.group(1).upper()
                    current_question["correct_option_id"] = ord(letter) - ord('A')
        
        # Add the last question
        if current_question and len(current_question.get("options", [])) == 4:
            questions.append(current_question)
        
        # Validate we have questions
        if not questions:
            raise ValueError("No questions generated")
        
        # Return requested number or available questions
        return questions[:num_questions]
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []

async def get_number(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send quiz polls based on user input."""
    try:
        num_questions = int(update.message.text)
        if num_questions <= 0:
            await update.message.reply_text("❌ Please enter a number greater than zero.")
            return ASK_NUM
        if num_questions > 10:
            await update.message.reply_text("⚠️ For best results, please request 10 questions or fewer.")
            return ASK_NUM

        text = context.user_data.get('text', '')
        if not text:
            await update.message.reply_text("❌ Oops! I lost your notes. Please start over with /start")
            return ConversationHandler.END

        msg = await update.message.reply_text(f"🧠 Generating {num_questions} questions for you...")

        try:
            questions = await generate_questions(text, num_questions)
        except Exception as e:
            print(f"Generation error: {e}")
            await msg.edit_text(f"❌ Error: {str(e)}")
            return ConversationHandler.END

        if not questions:
            await msg.edit_text("❌ Sorry, I couldn't generate questions. Please try different notes or fewer questions.")
            return ConversationHandler.END

        await msg.edit_text(f"✅ Generated {len(questions)} questions! Sending polls...")

        # Send each question as a quiz poll
        for i, q in enumerate(questions, 1):
            try:
                # Skip invalid questions
                if len(q.get("options", [])) != 4 or q.get("correct_option_id") not in [0, 1, 2, 3]:
                    print(f"Skipping invalid question: {q}")
                    continue
                    
                await context.bot.send_poll(
                    chat_id=update.effective_chat.id,
                    question=f"Q{i}: {q['question']}"[:300],  # Truncate to Telegram limit
                    options=q["options"],
                    type=Poll.QUIZ,
                    correct_option_id=q["correct_option_id"],
                    is_anonymous=False,
                    explanation=f"Correct answer: {q['options'][q['correct_option_id']][:200]}"
                )
                await asyncio.sleep(1)  # Prevent rate limiting
            except Exception as e:
                print(f"Error sending question {i}: {e}")

        await update.message.reply_text(
            "🎉 All questions sent!\n"
            "Use /start to create another quiz."
        )
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("❌ Please enter a valid number (1-10).")
        return ASK_NUM

def main():
    """Start the bot."""
    application = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_text)],
            ASK_NUM: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_number)],
        },
        fallbacks=[]
    )

    application.add_handler(conv_handler)

    print(f"Note2Poll bot is running with {MODEL_NAME} model! 🚀")
    application.run_polling()

if __name__ == '__main__':
    main()