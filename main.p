# main.py (Final Version: Dual API, Pixelation Game, Hint System, English Responses)

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
from PIL import Image, ImageFilter, Image
# Note: ImageFilter is not strictly needed for pixelation, but PIL is required.
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ChatPermissions
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ContextTypes, CallbackQueryHandler
)
from dotenv import load_dotenv
import time

# --- PostgreSQL Imports (Ensure psycopg2-binary is installed) ---
import psycopg2
from psycopg2 import sql
import psycopg2.extras 

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY") 

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
global_bot_status = True

# Store active games in memory 
# UPDATED Structure: {chat_id: {'answer': 'keyword', 'difficulty': 'easy', 'url': 'original_url', 'hints_taken': 0, 'hint_string': '_______'}}
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
    """Adds or deducts points from a user's score."""
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
# --- IMAGE GUESSING GAME LOGIC (DUAL API & PIXELATION) ---
# ------------------------------------------------------------------

# Pixelation Downscale Factors, Points, and Hint Logic
# Lower downscale factor means more pixelated (e.g., Extreme is 1/20th the size)
GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 5, 'hint_cost': 10},
    'hard': {'downscale_factor': 10, 'points': 70, 'max_hints': 5, 'hint_cost': 10},
    'medium': {'downscale_factor': 5, 'points': 50, 'max_hints': 5, 'hint_cost': 10},
    'easy': {'downscale_factor': 2, 'points': 30, 'max_hints': 5, 'hint_cost': 10}
}

search_queries = ['nature', 'city', 'animal', 'food', 'travel', 'object', 'landscape', 'mountain', 'beach', 'technology']

async def fetch_image_from_pexels(query: str):
    """Fetches image data from Pexels API."""
    if not PEXELS_API_KEY:
        raise Exception("PEXELS_API_KEY is missing.")
        
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=15&orientation=square"
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photos = response.json().get('photos', [])
    
    if not photos:
        raise Exception("Pexels: No photos found for query.")
        
    photo = random.choice(photos)
    image_url = photo['src']['large']
    # Pexels Answer: Use alt/photographer, fallback to search query
    raw_answer = photo.get('alt', photo.get('photographer', query)).split(',')[0].strip()
    
    return image_url, raw_answer, "Pexels"

async def fetch_image_from_unsplash(query: str):
    """Fetches image data from Unsplash API."""
    if not UNSPLASH_ACCESS_KEY:
        raise Exception("UNSPLASH_ACCESS_KEY is missing.")
        
    # Headers are preferred over query parameters for client_id
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    url = f"https://api.unsplash.com/photos/random?query={query}&orientation=squarish" 
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photo = response.json() 
    
    image_url = photo['urls']['regular']
    # Unsplash Answer: Use alt_description, fallback to description or search query
    raw_answer = photo.get('alt_description') or photo.get('description') or query
    
    return image_url, raw_answer, "Unsplash"

async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None]:
    """Randomly selects an API, fetches, and pixelates the image."""
    query = random.choice(search_queries)
    
    available_apis = []
    if PEXELS_API_KEY:
        available_apis.append(fetch_image_from_pexels)
    if UNSPLASH_ACCESS_KEY:
        available_apis.append(fetch_image_from_unsplash)

    if not available_apis:
        return None, "Both API Keys Missing. Check .env file.", None

    fetcher = random.choice(available_apis)

    try:
        # 1. Fetch Image URL and Raw Answer
        image_url, raw_answer, api_source = await fetcher(query)
        logger.info(f"Fetched image from {api_source} for query: {query}")

        # 2. Normalize and Clean the Answer
        # Use a regex to find all single words in the answer and pick a random one, or the search query
        all_words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_answer.lower())
        answer = random.choice(all_words) if all_words else query.lower()
        answer = answer.strip()
        
        # 3. Download Image Content
        image_data = requests.get(image_url, stream=True, timeout=10).content

        # 4. Process/Pixelate Image using PIL
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get pixelation factor
        downscale_factor = GAME_LEVELS[difficulty]['downscale_factor']
        
        # Calculate new size for downscaling (Pixelation effect)
        width, height = img.size
        # Ensure minimum size to avoid crashes
        if width < downscale_factor or height < downscale_factor:
            downscale_factor = min(width, height, 2)
            
        small_width = width // downscale_factor
        small_height = height // downscale_factor
        
        # Downscale (resample=NEAREST for blocky pixels)
        small_img = img.resize((small_width, small_height), Image.Resampling.NEAREST)
        
        # Scale back up to original size (maintains the pixelated look)
        pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)
        
        # Save pixelated image to BytesIO buffer
        output = io.BytesIO()
        pixelated_img.save(output, format='JPEG')
        output.seek(0)
        
        return output, answer, image_url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image API Request Error: {e}")
        return None, f"Image API Error: {e}", None
    except Exception as e:
        logger.error(f"Image Processing Error: {e}")
        return None, f"Image Processing/Unknown Error: {e}", None

# ------------------------------------------------------------------
# --- CORE BOT COMMANDS AND HANDLERS (ENGLISH RESPONSES) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name
    save_chat_id(chat_id)
    
    await update.message.reply_text(f"Hey there, **{user_name}**! 👋 I'm your core utility bot. Use /help to see my commands. ✨", parse_mode='Markdown')

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
        "✨ **Version**: 4.3 (Dual API, Pixelated Game)\n\n"
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
    """Sends a message to all known chats."""
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
# --- GAME COMMAND HANDLERS (UPDATED) ---
# ------------------------------------------------------------------

async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompts user to select game difficulty."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text("A game is already active! Try to guess it.")
        return

    # Display points clearly in buttons
    keyboard = [
        [
            InlineKeyboardButton(f"Extreme 🤯 (+{GAME_LEVELS['extreme']['points']} pts)", callback_data='game_extreme'),
            InlineKeyboardButton(f"Hard 🤔 (+{GAME_LEVELS['hard']['points']} pts)", callback_data='game_hard'),
        ],
        [
            InlineKeyboardButton(f"Medium 🧐 (+{GAME_LEVELS['medium']['points']} pts)", callback_data='game_medium'),
            InlineKeyboardButton(f"Easy 😎 (+{GAME_LEVELS['easy']['points']} pts)", callback_data='game_easy'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "**📸 Photo Guessing Game**\nSelect your difficulty level:", 
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles difficulty selection, hint request, and end game."""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    user_id = query.from_user.id
    data = query.data
    
    # --- END GAME LOGIC ---
    if data == 'game_end':
        game_state = active_games.pop(chat_id, None)
        if game_state:
            correct_answer = game_state['answer'].upper()
            original_url = game_state['url']
            
            # Use query.edit_message_caption if possible
            try:
                 await query.edit_message_caption(
                    caption=f"❌ **Game Over!** ❌\n You chose to end the game. The correct answer was: **{correct_answer}**.",
                    parse_mode='Markdown',
                    reply_markup=None
                )
            except:
                # If edit fails (e.g. too old), send a new message
                await context.bot.send_message(chat_id, "The original image is too old to edit. The game has ended.")
                
            # Send the original photo again if the caption couldn't be edited on the photo
            if 'The original image is too old to edit' in query.message.caption:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=f"The correct answer was: **{correct_answer}**.",
                    parse_mode='Markdown'
                )
            
        else:
            await context.bot.send_message(chat_id, "No active game to end.")
        return

    # --- HINT LOGIC ---
    if data == 'game_hint':
        game_state = active_games.get(chat_id)
        if not game_state:
            await context.bot.send_message(chat_id, "No active game to get a hint for.")
            return

        level_data = GAME_LEVELS[game_state['difficulty']]
        
        if game_state['hints_taken'] >= level_data['max_hints']:
            await context.bot.send_message(chat_id, "Maximum number of hints reached for this game.")
            return

        hint_cost = level_data['hint_cost']
        update_user_score(user_id, -hint_cost) # Deduct points
        game_state['hints_taken'] += 1
        
        answer_word = game_state['answer']
        current_hint_string = game_state['hint_string']
        
        # Find all masked indices
        masked_indices = [i for i, char in enumerate(current_hint_string) if char == '_']
        
        if masked_indices:
            # Randomly pick one masked letter to reveal
            reveal_index = random.choice(masked_indices)
            
            # Update hint string
            hint_list = list(current_hint_string)
            hint_list[reveal_index] = answer_word[reveal_index].upper()
            game_state['hint_string'] = "".join(hint_list)
        
        # Check if the whole word is revealed by hints
        if '_' not in game_state['hint_string']:
            # Force end the game if the word is fully revealed by hints (no points given)
            del active_games[chat_id]
            await context.bot.send_message(
                chat_id, 
                f"💡 **HINT TAKEN!** 💡\n-**{hint_cost} points** deducted.\n\nThe word **{answer_word.upper()}** has been fully revealed by hints. Game Over! No points awarded.",
                parse_mode='Markdown'
            )
            return


        new_score = get_user_score(user_id)
        
        # Game keyboard with updated hint count
        game_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"💡 Get Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - game_state['hints_taken']} left)", callback_data='game_hint')],
            [InlineKeyboardButton("🛑 End Game (Reveal Answer)", callback_data='game_end')]
        ])

        await context.bot.send_message(
            chat_id, 
            f"💡 **HINT TAKEN!** 💡\n-**{hint_cost} points** deducted. Current Score: **{new_score}**\n\n**Progress**: `{game_state['hint_string']}`\n",
            parse_mode='Markdown',
            reply_markup=game_keyboard
        )
        return

    # --- START GAME LOGIC ---
    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, "A game is already active!")
        return
    
    await query.edit_message_text(f"Game starting... Difficulty: **{difficulty.upper()}**", parse_mode='Markdown')
    
    # 1. Fetch and Pixelate Image
    pixelated_img_io, answer, original_url = await fetch_and_pixelate_image(difficulty)
    
    if not pixelated_img_io:
        await context.bot.send_message(chat_id, f"❌ Image could not be loaded. Error: {answer}")
        return
        
    # Initialize Hint String
    initial_hint_string = '_' * len(answer)
    
    # 2. Store Game State
    active_games[chat_id] = {
        'answer': answer,
        'difficulty': difficulty,
        'url': original_url,
        'hints_taken': 0,
        'hint_string': initial_hint_string
    }
    
    # 3. Send the Pixelated Image
    level_data = GAME_LEVELS[difficulty]
    points = level_data['points']
    
    caption = (
        f"**📸 Guessing Game: {difficulty.upper()}**\n"
        f"Guess this pixelated photo.\n"
        f"Correct answer yields **+{points} points**!\n"
        f"**Hint Progress**: `{initial_hint_string}`\n"
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Get Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints']} left)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 End Game (Reveal Answer)", callback_data='game_end')]
    ])
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='Markdown',
            reply_markup=game_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send pixelated photo: {e}")
        await context.bot.send_message(chat_id, "❌ Error sending the image. Game Cancelled.")
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
    
    # Check for keyword match
    if user_guess == correct_answer or user_guess in correct_answer.split() or correct_answer in user_guess.split():
        difficulty = game_state['difficulty']
        points = GAME_LEVELS[difficulty]['points']
        original_url = game_state['url']
        
        # 1. Update Score
        update_user_score(user_id, points)
        
        # 2. End Game
        del active_games[chat_id]
        
        # 3. Send original image and confirmation
        current_score = get_user_score(user_id)
        
        caption = (
            f"🎉 **CORRECT GUESS!** 🎉\n"
            f"**{user_name}** guessed **{correct_answer.upper()}**!\n"
            f"Difficulty: {difficulty.upper()} | Points Gained: **+{points}**\n"
            f"Your Total Score: **{current_score}**"
        )
        
        try:
            # Send the original image
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='Markdown'
            )
        except Exception:
            # Fallback if URL cannot be directly sent (rare)
            await context.bot.send_message(chat_id, f"{caption}\n(Error sending original photo).", parse_mode='Markdown')
            
async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user's current score."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    score = get_user_score(user_id)
    await update.message.reply_text(
        f"**{user_name}**! Your current score is: **{score}** points! 🏅",
        parse_mode='Markdown'
    )

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the top 10 scores."""
    top_scores = get_top_scores(10)
    
    if not top_scores:
        await update.message.reply_text("The Leaderboard is empty. Play a game to earn scores!")
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
    
    if not PEXELS_API_KEY and not UNSPLASH_ACCESS_KEY:
        logger.critical("CRITICAL: Both PEXELS_API_KEY and UNSPLASH_ACCESS_KEY are missing. Image guessing game will not work.")

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

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION), 
        process_message
    ))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Callback Query Handler for the game (difficulty, hint, and end game)
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
    # Ensure you have installed all dependencies: pip install -r requirements.txt
    # And updated your .env file with TELEGRAM_BOT_TOKEN, DATABASE_URL, PEXELS_API_KEY, UNSPLASH_ACCESS_KEY, and BROADCAST_ADMIN_ID
    main()
