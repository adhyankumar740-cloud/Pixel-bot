# main_core_broadcast_game.py (PostgreSQL, Broadcast, and Image Guessing Game)
import os
import logging
import requests
import asyncio
import uuid
import pytz
import traceback
import random
from collections import defaultdict
from datetime import datetime
import psutil
import json
import re
import urllib.parse
import io
# --- PostgreSQL Imports ---
import psycopg2
from psycopg2 import sql
import psycopg2.extras 
# --- Image Processing & API Imports ---
from PIL import Image, ImageFilter
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ChatPermissions
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ContextTypes, CallbackQueryHandler
)
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# BROADCAST_ADMIN_ID is required for the broadcast function
try:
    BROADCAST_ADMIN_ID = int(os.getenv("BROADCAST_ADMIN_ID"))
except (ValueError, TypeError):
    BROADCAST_ADMIN_ID = 0
    logging.error("CRITICAL: BROADCAST_ADMIN_ID is missing or not a valid number. Broadcast functionality will be disabled.")

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Global Game & Stats Variables ---
start_time = datetime.now()
total_messages_processed = 0
known_users = set() 
global_bot_status = True # Kept, though control commands are removed

# Store active games in memory 
active_games = defaultdict(dict) 

# --- Global DB connection variable ---
conn = None

# --- Logging Basic Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# --- POSTGRESQL/NEON DB CONNECTION AND DATA MANAGEMENT FUNCTIONS ---
# ------------------------------------------------------------------

def db_connect():
    """Establishes and returns a PostgreSQL connection, and creates necessary tables."""
    global conn
    if conn and not conn.closed:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return conn, None
        except Exception:
            logger.warning("Existing DB connection failed, attempting to reconnect.")
            conn = None 

    if not DATABASE_URL:
        return None, "DATABASE_URL environment variable is missing."

    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        
        # Initialize tables
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS known_chats (
                    chat_id BIGINT PRIMARY KEY,
                    joined_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS game_scores (
                    user_id BIGINT PRIMARY KEY,
                    score INT DEFAULT 0
                );
            """)
        
        logger.info("Successfully connected to PostgreSQL (Neon Tech) and verified tables.")
        return conn, None
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        conn = None 
        return None, f"Error connecting to PostgreSQL: {e}"

def load_known_users():
    """Loads all chat IDs for broadcast from the DB."""
    global known_users
    conn, error = db_connect()
    if error:
        logger.error(f"Could not load known users: {error}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM known_chats;")
            known_users = {str(row[0]) for row in cur.fetchall()}
        logger.info(f"Loaded {len(known_users)} chats from DB.")
    except Exception as e:
        logger.error(f"Error loading known users from DB: {e}")

def save_chat_id(chat_id):
    """Saves a new chat ID to the DB."""
    conn, error = db_connect()
    if error:
        logger.error(f"Could not save chat ID: {error}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO known_chats (chat_id) VALUES (%s) ON CONFLICT (chat_id) DO NOTHING;",
                (chat_id,)
            )
    except Exception as e:
        logger.error(f"Error saving chat ID to DB: {e}")

# --- Game Score Functions (UNCHANGED) ---

def get_user_score(user_id: int) -> int:
    """Retrieves a user's current game score."""
    conn, error = db_connect()
    if error:
        logger.error(f"DB Error for score: {error}")
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT score FROM game_scores WHERE user_id = %s;", (user_id,))
            result = cur.fetchone()
            return result[0] if result else 0
    except Exception as e:
        logger.error(f"Error fetching score for user {user_id}: {e}")
        return 0

def update_user_score(user_id: int, points: int):
    """Adds points to a user's score."""
    conn, error = db_connect()
    if error:
        logger.error(f"DB Error for score update: {error}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO game_scores (user_id, score) VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET score = game_scores.score + EXCLUDED.score;
                """,
                (user_id, points)
            )
        logger.info(f"Updated score for user {user_id} by {points} points.")
    except Exception as e:
        logger.error(f"Error updating score for user {user_id}: {e}")

def get_top_scores(limit: int = 10):
    """Fetches the top scores from the DB."""
    conn, error = db_connect()
    if error:
        logger.error(f"DB Error for leaderboard: {error}")
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, score FROM game_scores ORDER BY score DESC LIMIT %s;", (limit,))
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        return []

# ------------------------------------------------------------------
# --- IMAGE GUESSING GAME LOGIC (UNCHANGED) ---
# ------------------------------------------------------------------

BLUR_LEVELS = {
    'extreme': {'radius': 20, 'points': 500},
    'hard': {'radius': 12, 'points': 300},
    'medium': {'radius': 6, 'points': 150},
    'easy': {'radius': 3, 'points': 50}
}

async def fetch_and_blur_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None]:
    """Fetches a random image from Pexels, blurs it, and returns the blurred image, answer, and original URL."""
    if not PEXELS_API_KEY:
        logger.error("PEXELS_API_KEY is not set.")
        return None, "API Key Missing", None

    search_queries = ['nature', 'city', 'animal', 'food', 'travel', 'object', 'landscape']
    query = random.choice(search_queries)
    
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=15&orientation=square"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        photos = response.json().get('photos', [])
        
        if not photos:
            return None, "No photos found.", None
            
        photo = random.choice(photos)
        image_url = photo['src']['large']
        raw_answer = photo.get('alt', photo.get('photographer', query)).split(',')[0].strip()
        answer = re.sub(r'[^a-zA-Z\s]', '', raw_answer).lower().strip()
        
        if not answer or len(answer) < 3:
             answer = query 

        image_data = requests.get(image_url, stream=True, timeout=10).content

        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        blur_radius = BLUR_LEVELS[difficulty]['radius']
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        output = io.BytesIO()
        blurred_img.save(output, format='JPEG')
        output.seek(0)
        
        return output, answer, image_url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image API Request Error: {e}")
        return None, f"Image API Error: {e}", None
    except Exception as e:
        logger.error(f"Image Processing Error: {e}")
        return None, f"Image Processing Error: {e}", None

# ------------------------------------------------------------------
# --- CORE BOT COMMANDS AND HANDLERS ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name
    save_chat_id(chat_id)
    
    await update.message.reply_text(f"Hey there, {user_name}! 👋 I'm your core utility bot. Use /help to see my commands. ✨")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the simplified help message."""
    
    help_text = (
        "✨ **Bot Commands List** ✨\n\n"
        "**🤖 Core Commands**\n"
        "`/start` - Start the bot (Private chat).\n"
        "`/help` - Show this message.\n"
        "`/about` - Know more about the bot.\n"
        "`/stats` - Show bot's health and uptime.\n"
        "`/id` - Get your User ID and Chat ID.\n\n"
        
        "**📸 Image Guessing Game**\n"
        "`/game` - Start a new image guessing game and select difficulty.\n"
        "`/myscore` - Check your current game score.\n"
        "`/leaderboard` - Show the top 10 players.\n\n"
        
        "**👑 Owner Command**\n"
        "`/broadcast <message>` - Send a message to all known chats. (Only for the Owner)\n\n"
        
        "✨ by Adhyan ✨"
    )
        
    await update.message.reply_text(help_text, parse_mode='Markdown', disable_web_page_preview=True)
    logger.info(f"[{update.effective_chat.id}] /help command used.")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    about_text = (
        "❤️ **About Bot** ❤️\n\n"
        "I am a custom core utility bot, streamlined to focus on **Broadcast** and the **Image Guessing Game**.\n\n"
        "✨ **Creator**: Adhyan\n"
        "✨ **Version**: 4.1 (Clean Core)\n\n"
    )
    await update.message.reply_text(about_text, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uptime = datetime.now() - start_time
    stats_text = (
        f"📊 **Bot Status**\n"
        f"**Uptime**: {str(uptime).split('.')[0]}\n"
        f"**Known Chats**: {len(known_users)}\n"
        f"**Status**: {'✅ ON' if global_bot_status else '❌ OFF'}\n"
    )
    await update.message.reply_text(stats_text, parse_mode='Markdown')

async def get_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows user and chat ID."""
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    text = (
        f"**Your ID**: `{user.id}`\n"
        f"**Chat ID**: `{chat_id}`"
    )
    
    if update.message.reply_to_message:
        replied_user = update.message.reply_to_message.from_user
        text += f"\n**Replied User ID**: `{replied_user.id}`"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error."""
    logger.error("An error occurred:", exc_info=context.error)
    # Removed error reporting to admin for simplicity (as admin commands were removed)


# ------------------------------------------------------------------
# --- OWNER COMMANDS (Only Broadcast) ---
# ------------------------------------------------------------------

def check_owner_wrapper(handler):
    """Decorator to restrict access to the primary bot owner (BROADCAST_ADMIN_ID)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if user_id == str(BROADCAST_ADMIN_ID):
            await handler(update, context)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, this command is restricted to the bot owner only.")
    return wrapper

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message to all known chats (KEPT)."""
    # Logic remains the same as it was the only command the user wanted to keep
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Please provide a message to broadcast. Usage: `/broadcast <message>`")
        return
    
    message = " ".join(context.args)
    sent_count = 0
    failed_count = 0
    
    await update.message.reply_text(f"Starting broadcast to {len(known_users)} chats...")
    logger.info(f"[{user_id}] Initiating broadcast: {message}")

    chats_to_broadcast = list(known_users)
    
    for chat_id_str in chats_to_broadcast:
        chat_id = int(chat_id_str)
        try:
            await context.bot.send_message(
                chat_id=chat_id, 
                text=message, 
                parse_mode='Markdown'
            )
            sent_count += 1
            await asyncio.sleep(0.1) 
        except Exception as e:
            failed_count += 1
            logger.warning(f"Broadcast failed for chat {chat_id}: {e}")

    final_message = f"**Broadcast finished!**\nSent successfully to: `{sent_count}` chats.\nFailed for: `{failed_count}` chats."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=final_message, parse_mode='Markdown')
    logger.info(f"[{user_id}] Broadcast completed. Sent: {sent_count}, Failed: {failed_count}")


# ------------------------------------------------------------------
# --- GAME COMMAND HANDLERS (UNCHANGED) ---
# ------------------------------------------------------------------

async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompts user to select game difficulty."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text("Ek game pehle se hi chal raha hai! Koshish karo usse guess karne ki.")
        return

    keyboard = [
        [
            InlineKeyboardButton("Extreme 🤯", callback_data='game_extreme'),
            InlineKeyboardButton("Hard 🤔", callback_data='game_hard'),
        ],
        [
            InlineKeyboardButton("Medium 🧐", callback_data='game_medium'),
            InlineKeyboardButton("Easy 😎", callback_data='game_easy'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "**Photo Guessing Game**\nKhelne ke liye mushkil (difficulty) chuno:", 
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles difficulty selection and starts the game."""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    data = query.data
    
    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, "Ek game pehle se hi chal raha hai!")
        return
    
    await query.edit_message_text(f"Game shuru ho raha hai... Difficulty: **{difficulty.upper()}**", parse_mode='Markdown')
    
    blurred_img_io, answer, original_url = await fetch_and_blur_image(difficulty)
    
    if not blurred_img_io:
        await context.bot.send_message(chat_id, f"❌ Image load nahi ho payi. Error: {answer}")
        return
        
    active_games[chat_id] = {
        'answer': answer,
        'difficulty': difficulty,
        'url': original_url
    }
    
    points = BLUR_LEVELS[difficulty]['points']
    caption = (
        f"**📸 Guessing Game: {difficulty.upper()}**\n"
        f"Iss blurred photo ko guess karo.\n"
        f"Sahi jawab par **{points} points** milenge!\n"
        f"**Hint**: Jawab ek single word ho sakta hai jo photo ke object, location ya category ko describe karta ho."
    )
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=blurred_img_io, 
            caption=caption, 
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to send blurred photo: {e}")
        await context.bot.send_message(chat_id, "❌ Blurred photo bhejte samay error aa gaya. Game Cancelled.")
        del active_games[chat_id]

async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Checks the user's guess against the active game's answer."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    game_state = active_games.get(chat_id)
    if not game_state:
        return 

    correct_answer = game_state['answer'].lower()
    user_guess = text.lower().strip()
    
    if user_guess == correct_answer or user_guess in correct_answer.split() or correct_answer in user_guess:
        difficulty = game_state['difficulty']
        points = BLUR_LEVELS[difficulty]['points']
        original_url = game_state['url']
        
        update_user_score(user_id, points)
        del active_games[chat_id]
        
        current_score = get_user_score(user_id)
        
        caption = (
            f"🎉 **SAHI JAWAB!** 🎉\n"
            f"**{user_name}** ne **{correct_answer.upper()}** guess kiya!\n"
            f"Difficulty: {difficulty.upper()} | Points Gained: **+{points}**\n"
            f"Aapka Total Score: **{current_score}**"
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='Markdown'
            )
        except Exception:
            await context.bot.send_message(chat_id, f"{caption}\n(Original Photo bhejte samay error aaya).", parse_mode='Markdown')
            
async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user's current score."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    score = get_user_score(user_id)
    await update.message.reply_text(
        f"**{user_name}**! Aapka current score hai: **{score}** points! 🏅",
        parse_mode='Markdown'
    )

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the top 10 scores."""
    top_scores = get_top_scores(10)
    
    if not top_scores:
        await update.message.reply_text("Leaderboard khali hai. Koi game khelo aur score banao!")
        return
        
    leaderboard_text = "🏆 **Global Leaderboard (Top 10)** 🏆\n\n"
    
    for i, (user_id, score) in enumerate(top_scores, 1):
        try:
            # Fetch user info for name
            user = await context.bot.get_chat(user_id)
            user_name = user.full_name
        except Exception:
            user_name = f"User {user_id}"

        leaderboard_text += f"{i}. **{user_name}** - `{score}` points\n"
        
    await update.message.reply_text(leaderboard_text, parse_mode='Markdown')

# ------------------------------------------------------------------
# --- MESSAGE HANDLERS (Minimal) ---
# ------------------------------------------------------------------

async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Saves the chat ID when the bot is added to a new group."""
    chat_id = update.effective_chat.id
    for member in update.effective_message.new_chat_members:
        if member.id == context.bot.id:
            save_chat_id(chat_id)
            logger.info(f"Bot added to new chat, ID saved: {chat_id}")
            break

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text messages (for game guessing)."""
    global total_messages_processed
    total_messages_processed += 1
    
    text = update.effective_message.text or update.effective_message.caption
    
    if not text:
        return
        
    await check_guess_and_update_score(update, context, text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice handler (Now just a placeholder)."""
    pass

# ------------------------------------------------------------------
# --- MAIN EXECUTION ---
# ------------------------------------------------------------------

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # 1. Connect to DB and load data
    conn, error = db_connect()
    if not conn:
        logger.critical(f"CRITICAL: Failed to connect to PostgreSQL on startup. {error}. Check your DATABASE_URL.")
    
    load_known_users()
    
    if not PEXELS_API_KEY:
        logger.critical("CRITICAL: PEXELS_API_KEY is missing. Image guessing game will not work.")

    # 2. Add Handlers
    
    # Core Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("id", get_id_command))

    # Game Commands
    application.add_handler(CommandHandler("game", game_command))
    application.add_handler(CommandHandler("myscore", my_score_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))

    # Owner-Only Command
    application.add_handler(CommandHandler("broadcast", check_owner_wrapper(broadcast_command))) 

    # --- REMOVED: poweron, poweroff, adminstats, showchats, s_ban, s_unban, s_send, addsudo, removesudo ---

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION), 
        process_message
    ))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Callback Query Handler for the game
    application.add_handler(CallbackQueryHandler(handle_game_callback))

    application.add_error_handler(error_handler)
    
    # 3. Start the bot
    if WEBHOOK_URL:
        PORT = int(os.getenv("PORT", "8000"))
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
        )
        logger.info(f"Bot started with webhook on port {PORT}")
    else:
        logger.info("Bot started with polling")
        application.run_polling(poll_interval=1)

if __name__ == '__main__':
    main()
