# main.py (Final, Complete, Standardized, Flow Fixed, Rapid Quiz Renamed, MarkdownV2 Escaping Fully Fixed)

import os
import logging
import requests
import asyncio
import uuid
import pytz
import traceback
import random
from collections import defaultdict
from datetime import datetime, timedelta
import psutil
import json
import re
import urllib.parse
import io
from PIL import Image

# --- GEMINI INTEGRATION ---
try:
    import google.genai
    from google.genai import types
except ImportError:
    print("NOTE: 'google-genai' library required for AI features.")

# --- PostgreSQL Imports (Ensure psycopg2-binary is installed) ---
import psycopg2
from psycopg2 import sql
import psycopg2.extras 

# --- Telegram Imports ---
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
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# Store active games in memory (Normal Mode - Single Round)
active_games = defaultdict(dict) 

# Store rapid-quiz games (Multi-Round, Timed)
rapid_games = defaultdict(dict) 

# --- Global DB connection variable ---
conn = None

# --- Logging Basic Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# --- CONSTANTS & CONFIGURATIONS ---
# ------------------------------------------------------------------

BOT_USERNAME = "Pixel Peep" 
SUPPORT_GROUP_LINK = "https://t.me/pixel_studis" 
MAIN_GROUP_LINK = "https://t.me/+JNpAXIuwHTIyOWY1" 
WELCOME_IMAGE_URL = "AgACAgUAAxkBAAPNaSdU6DYWrJbMvkAsoRF93H3V2x8AAhYLaxsfjEFVReKLNrpOrBUBAAMCAAN4AAM2BA" 
DONATION_QR_CODE_ID = "AgACAgUAAxkBAAIBzGkn8CZvfaDPAckxv-cOPFgKlus4AAJdC2sbmhxBVWBRzMvm7w0HAQADAgADeQADNgQ" 

GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 3, 'hint_cost': 10}, 
    'hard': {'downscale_factor': 15, 'points': 70, 'max_hints': 3, 'hint_cost': 10}, 
    'medium': {'downscale_factor': 10, 'points': 50, 'max_hints': 3, 'hint_cost': 10}, 
    'easy': {'downscale_factor': 25, 'points': 30, 'max_hints': 3, 'hint_cost': 10} 
}

SEARCH_CATEGORIES = {
    'nature': '🌳', 'city': '🏙️', 'animal': '🐾', 'food': '🍕', 
    'travel': '✈️', 'object': '💡', 'landscape': '🏞️', 'mountain': '⛰️', 
    'beach': '🏖️', 'technology': '🤖', 'vintage': '🕰️', 'sports': '⚽', 
    'art': '🎨', 'music': '🎶', 'architecture': '🏛️', 'car': '🚗',
    'flower': '🌸', 'instrument': '🎸', 'furniture': '🛋️', 'clothing': '👕',
    'shoes': '👟', 'coffee': '☕', 'dessert': '🍰', 'tree': '🌲', 
    'river': '🌊', 'sky': '☁️', 'space': '🚀', 'building': '🏢', 
    'street': '🛣️', 'bridge': '🌉', 'train': '🚂', 'boat': '⛵',
    'fruit': '🍎', 'vegetable': '🥕', 'drink': '🍹', 'tool': '🔨',
    'toy': '🧸', 'kitchen': '🔪', 'light': '💡', 'shadow': '👤',
    'abstract': '🌀', 'geometry': '🔺', 'texture': '🧱', 'pattern': '🖼️',
    'wildlife': '🐅', 'bird': '🐦', 'fish': '🐠', 'reptile': '🦎',
    'desert': '🏜️', 'jungle': '🌿', 'snow': '❄️', 'fire': '🔥',
    'waterfall': '💧', 'cave': '🕳️', 'library': '📚', 'museum': '🖼️',
    'hospital': '🏥', 'school': '🏫', 'office': '💼', 'factory': '🏭',
    'money': '💰', 'jewellery': '💎', 'watch': '⌚', 'computer': '💻',
    'phone': '📱', 'camera': '📷', 'robot': '🤖', 'drone': '🚁',
    'garden': '🪴', 'farm': '🚜', 'harvest': '🌾', 'wine': '🍷',
    'bread': '🍞', 'cheese': '🧀', 'meat': '🍖', 'seafood': '🍤',
    'sunrise': '🌅', 'sunset': '🌇', 'night': '🌃', 'rain': '🌧️',
    'fog': '🌫️', 'ice': '🧊', 'statue': '🗿', 'fountain': '⛲',
    'mask': '🎭', 'gloves': '🧤', 'hat': '👒', 'bag': '👜',
    'key': '🔑', 'lock': '🔒', 'door': '🚪', 'window': '🪟', 'road': '🛣️',
    'mountain_peak': '🗻', 'volcano': '🌋', 'island': '🏝️', 'field': '🟢', 
    'microphone': '🎤', 'headphones': '🎧', 'gaming': '🎮', 'book': '📖'
}
search_queries = list(SEARCH_CATEGORIES.keys())

SHOP_ITEMS = {
    '1': {'name': 'Flashlight', 'cost': 200, 'description': 'Reveal 2 random unrevealed letters in the word.'},
    '2': {'name': 'First Letter', 'cost': 100, 'description': 'Reveal the first letter of the answer.'}
}

DAILY_BONUS_POINTS = 50

RAPID_QUIZ_SETTINGS = {
    'difficulty': 'medium', 
    'reward': 20,           
    'max_rounds': 10,
    'BASE_TIME': 50,
    'TIME_DECREMENT': 5,
    'MIN_TIME': 10
}

# ------------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# ------------------------------------------------------------------

def escape_markdown_v2(text: str) -> str:
    """Escapes Telegram MarkdownV2 special characters."""
    # Escape characters: '_*[]()~`>#+-=|{}.!'
    special_chars = r'_*[]()~`>#+-=|{}.!'
    # Use re.sub with a raw string for the replacement pattern to handle backslashes correctly
    return re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', text)

def get_rapid_quiz_time_limit(round_number: int) -> int:
    """Calculates the time limit based on the round number (50, 45, 40, 35, ... min 10)."""
    settings = RAPID_QUIZ_SETTINGS
    time_limit = settings['BASE_TIME'] - (round_number - 1) * settings['TIME_DECREMENT']
    return max(settings['MIN_TIME'], time_limit)

def get_difficulty_menu():
    """Returns the InlineKeyboardMarkup for selecting game difficulty."""
    keyboard = [
        [
            InlineKeyboardButton(f"🐣 Easy (+{GAME_LEVELS['easy']['points']} pts)", callback_data='game_easy'),
            InlineKeyboardButton(f"🧘 Medium (+{GAME_LEVELS['medium']['points']} pts)", callback_data='game_medium')
        ],
        [
            InlineKeyboardButton(f"💪 Hard (+{GAME_LEVELS['hard']['points']} pts)", callback_data='game_hard'),
            InlineKeyboardButton(f"💀 Extreme (+{GAME_LEVELS['extreme']['points']} pts)", callback_data='game_extreme')
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def send_difficulty_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends the difficulty selection menu, handling both commands and callbacks."""
    # Standardized and Escaped Text
    text = r"**🖼️ C H O O S E   C H A L L E N G E**\n\nSelect a precision level for the image analysis \(Difficulty\)\:"
    reply_markup = get_difficulty_menu()
    
    chat_id = update.effective_chat.id if update.effective_chat else (update.callback_query.message.chat_id if update.callback_query else None)
    if not chat_id: return

    if update.callback_query:
        query = update.callback_query
        try:
            await query.answer()
            await query.edit_message_text(text, parse_mode='MarkdownV2', reply_markup=reply_markup)
        except Exception:
            await context.bot.send_message(chat_id, text, parse_mode='MarkdownV2', reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text, parse_mode='MarkdownV2', reply_markup=reply_markup)

# ------------------------------------------------------------------
# --- DATABASE CONNECTION AND DATA MANAGEMENT FUNCTIONS ---
# ------------------------------------------------------------------

def db_connect():
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
        
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS known_chats (
                    chat_id BIGINT PRIMARY KEY,
                    joined_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS game_scores (
                    user_id BIGINT PRIMARY KEY,
                    score INT DEFAULT 0,
                    solved_count INT DEFAULT 0,
                    current_streak INT DEFAULT 0,
                    last_daily_claim TIMESTAMP WITH TIME ZONE DEFAULT NULL
                );
                CREATE TABLE IF NOT EXISTS user_collection (
                    user_id BIGINT,
                    category VARCHAR(50),
                    UNIQUE (user_id, category)
                );
            """)

            try:
                cur.execute("ALTER TABLE game_scores ADD COLUMN solved_count INT DEFAULT 0;")
            except Exception:
                pass
            
            try:
                cur.execute("ALTER TABLE game_scores ADD COLUMN current_streak INT DEFAULT 0;")
            except Exception:
                pass
            
            try:
                cur.execute("ALTER TABLE game_scores ADD COLUMN last_daily_claim TIMESTAMP WITH TIME ZONE DEFAULT NULL;")
            except Exception:
                pass
        
        logger.info("Successfully connected to PostgreSQL (Neon Tech) and verified tables.")
        return conn, None
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        conn = None 
        return None, f"Error connecting to PostgreSQL: {e}"

def load_known_users():
    global known_users
    conn, error = db_connect()
    if error:
        logger.error(f"Could not load known users: {error}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM known_chats;")
            known_users = {str(row[0]) for row in cur.fetchall()}
    except Exception as e:
        logger.error(f"Error loading known users from DB: {e}")

def save_chat_id(chat_id):
    conn, error = db_connect()
    if error: return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO known_chats (chat_id) VALUES (%s) ON CONFLICT (chat_id) DO NOTHING;",
                (chat_id,)
            )
    except Exception as e:
        logger.error(f"Error saving chat ID to DB: {e}")

def get_user_score(user_id: int) -> int:
    conn, error = db_connect()
    if error: return 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT score FROM game_scores WHERE user_id = %s;", (user_id,))
            result = cur.fetchone()
            return result[0] if result else 0
    except Exception as e:
        logger.error(f"Error fetching score for user {user_id}: {e}")
        return 0

def update_user_score(user_id: int, points: int):
    conn, error = db_connect()
    if error: return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO game_scores (user_id, score, solved_count, current_streak) 
                VALUES (%s, %s, 0, 0)
                ON CONFLICT (user_id) 
                DO UPDATE SET score = game_scores.score + EXCLUDED.score;
                """,
                (user_id, points)
            )
    except Exception as e:
        logger.error(f"Error updating score for user {user_id}: {e}")

def get_top_scores(limit: int = 10):
    conn, error = db_connect()
    if error: return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, score FROM game_scores ORDER BY score DESC LIMIT %s;", (limit,))
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        return []

async def get_user_profile_data(user_id: int):
    conn, error = db_connect()
    if error: return None, 0, 0, None, "DB Error"
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT score, solved_count, current_streak, last_daily_claim FROM game_scores WHERE user_id = %s;",
                (user_id,)
            )
            data = cur.fetchone()
            if not data:
                try:
                    cur.execute(
                        "INSERT INTO game_scores (user_id, score) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING;",
                        (user_id,)
                    )
                except Exception as insert_e:
                    logger.error(f"Error initializing user profile: {insert_e}")
                    
                data = (0, 0, 0, datetime.now(pytz.utc) - timedelta(days=2))
            
            score, solved_count, current_streak, last_daily_claim = data

            cur.execute("SELECT COUNT(*) FROM game_scores WHERE score > %s;", (score,))
            rank = cur.fetchone()[0] + 1
            
            return rank, score, solved_count, current_streak, last_daily_claim
    except Exception as e:
        logger.error(f"Error fetching profile data for user {user_id}: {e}")
        return None, 0, 0, 0, None

async def update_daily_claim(user_id: int, points: int):
    conn, error = db_connect()
    if error: return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO game_scores (user_id, score, last_daily_claim) 
                VALUES (%s, %s, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET score = game_scores.score + %s, last_daily_claim = NOW();
                """,
                (user_id, points, points)
            )
        return True
    except Exception as e:
        logger.error(f"Error updating daily claim for user {user_id}: {e}")
        return False

async def get_user_collection(user_id: int):
    conn, error = db_connect()
    if error: return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT category FROM user_collection WHERE user_id = %s ORDER BY category ASC;", (user_id,))
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching collection for user {user_id}: {e}")
        return []

async def save_solved_image(user_id: int, category: str):
    conn, error = db_connect()
    if error: return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_collection (user_id, category) VALUES (%s, %s) ON CONFLICT (user_id, category) DO NOTHING;",
                (user_id, category)
            )

            cur.execute(
                """
                UPDATE game_scores SET 
                    solved_count = solved_count + 1,
                    current_streak = current_streak + 1
                WHERE user_id = %s;
                """,
                (user_id,)
            )
    except Exception as e:
        logger.error(f"Error saving solved image data for user {user_id}: {e}")


# ------------------------------------------------------------------
# --- GEMINI AI INTEGRATION AND UTILITY FUNCTIONS ---
# ------------------------------------------------------------------

def initialize_gemini_client():
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. AI functions will fail.")
        return None
    try:
        client = google.genai.Client()
        return client
    except Exception as e:
        logger.error(f"Gemini Client Initialization Error: {e}")
        return None

async def get_ai_answer_and_hint(image_data: bytes) -> tuple[str | None, str | None]:
    client = initialize_gemini_client()
    if client is None:
        return None, None
        
    image_part = types.Part.from_bytes(
        data=image_data,
        mime_type='image/jpeg'
    )

    prompt = (
        "Analyze this photograph. Your task is to provide: "
        "1. A single, specific, common noun or object name as the **answer**. The word should contain only letters (no spaces or hyphens) and be 3 to 15 characters long. "
        "2. A single, short, descriptive sentence as a **hint** for that answer. "
        "Strictly return ONLY a JSON object with two keys: 'answer' (one word, lowercase) and 'hint' (one sentence)."
    )
    
    response = None
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[image_part, prompt]
        )
        
        json_text = response.text.strip().replace("```json", "").replace("```", "").replace("`", "").strip()
        result = json.loads(json_text)
        
        one_word_answer = re.sub(r'[^a-z]', '', result.get('answer', '').lower().strip())
        hint_sentence = result.get('hint', 'The AI failed to generate a descriptive clue.')
        
        if not one_word_answer or len(one_word_answer) < 3:
             logger.warning(f"AI generated invalid answer: {one_word_answer}. Falling back.")
             return None, None
             
        return one_word_answer, hint_sentence
        
    except Exception as e:
        logger.error(f"AI Response or Parsing Error: {e} | Raw Text: {response.text if response else 'N/A'}")
        return None, None

# ------------------------------------------------------------------
# --- IMAGE GUESSING GAME LOGIC (API Interaction) ---
# ------------------------------------------------------------------

async def fetch_image_from_pexels(query: str):
    if not PEXELS_API_KEY:
        raise Exception("PEXELS_API_KEY is missing.")
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page=15&orientation=square"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photos = response.json().get('photos', [])
    if not photos:
        raise Exception("Pexels: No photos found for query.")
    photo = random.choice(photos)
    image_url = photo['src']['large']
    return image_url, query, "Pexels" 

async def fetch_image_from_unsplash(query: str):
    if not UNSPLASH_ACCESS_KEY:
        raise Exception("UNSPLASH_ACCESS_KEY is missing.")
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    url = f"https://api.unsplash.com/photos/random?query={urllib.parse.quote(query)}&orientation=squarish" 
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photo = response.json() 
    image_url = photo['urls']['regular']
    return image_url, query, "Unsplash"

async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None, str | None, str | None]:
    category = random.choice(search_queries)
    query = category
    
    available_apis = []
    if PEXELS_API_KEY:
        available_apis.append(fetch_image_from_pexels)
    if UNSPLASH_ACCESS_KEY:
        available_apis.append(fetch_image_from_unsplash)

    if not available_apis:
        return None, "Both API Keys Missing. Check .env file.", None, None, None

    image_url = None
    for fetcher in available_apis:
        try:
            image_url, _, api_source = await fetcher(query)
            if image_url:
                logger.info(f"Fetched image from {api_source} for query: {query}")
                break
        except Exception as e:
            logger.error(f"Fetcher error: {e}")

    if not image_url:
        return None, "Error: Could not fetch image from APIs.", None, None, None

    try:
        image_response = requests.get(image_url, stream=True, timeout=10)
        image_response.raise_for_status()
        image_data = image_response.content
        
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        AI_MAX_SIZE = (512, 512)
        ai_input_img = img.copy()
        ai_input_img.thumbnail(AI_MAX_SIZE) 
        
        ai_input_bytes = io.BytesIO()
        ai_input_img.save(ai_input_bytes, format='JPEG', quality=85) 
        ai_input_bytes.seek(0)

        ai_answer, hint_sentence = await get_ai_answer_and_hint(ai_input_bytes.read()) 
        
        if not ai_answer:
            return None, "AI failed to generate a valid, one-word answer.", None, None, None
        
        downscale_factor = GAME_LEVELS[difficulty]['downscale_factor']
        
        width, height = img.size
        
        if width < downscale_factor or height < downscale_factor:
            downscale_factor = min(width, height, 2)
            
        if isinstance(downscale_factor, float):
            downscale_factor = max(1.0, downscale_factor) 
            small_width = int(width / downscale_factor)
            small_height = int(height / downscale_factor)
        else:
            small_width = width // downscale_factor
            small_height = height // downscale_factor

        small_width = max(1, small_width)
        small_height = max(1, small_height)
        
        small_img = img.resize((small_width, small_height), Image.Resampling.NEAREST)
        pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)
        
        output = io.BytesIO()
        pixelated_img.save(output, format='JPEG')
        output.seek(0)
        
        return output, ai_answer, image_url, category, hint_sentence
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image Download Error: {e}")
        return None, f"Image Download Error: {e}", None, None, None
    except Exception as e:
        logger.error(f"Image Processing/Unknown Error: {e}")
        return None, f"Image Processing/Unknown Error: {e}", None, None, None

# ------------------------------------------------------------------
# --- RAPID QUIZ GAME FUNCTIONS ---
# ------------------------------------------------------------------

async def timeout_rapid_round_task(chat_id: int, context: ContextTypes.DEFAULT_TYPE, time_limit: int):
    try:
        await asyncio.sleep(time_limit)
        
        if chat_id not in rapid_games:
            return
            
        state = rapid_games[chat_id]
        
        try:
            if 'last_message_id' in state:
                # Standardized Caption for Time Up
                await context.bot.edit_message_caption(
                    chat_id=chat_id, 
                    message_id=state['last_message_id'],
                    caption=rf"⏱️ **T I M E   U P** ⏱️\n\n**Round {state['current_round']} Skipped\!** The answer was **{state['answer'].upper()}**\.",
                    reply_markup=None,
                    parse_mode='MarkdownV2' 
                )
        except Exception:
            pass 

        await context.bot.send_message(
            chat_id=chat_id, 
            text=rf"⏱️ **T I M E   U P** ⏱️\n\n**Round {state['current_round']} Skipped\!** The answer was **{state['answer'].upper()}**\.",
            parse_mode='MarkdownV2' 
        )
        
        state['current_round'] += 1
        
        if state['current_round'] > state['max_rounds']:
            await end_rapid_quiz_logic(chat_id, context)
        else:
            await start_rapid_round(chat_id, context)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in timeout_rapid_round_task for chat {chat_id}: {e}")
        if chat_id in rapid_games:
            await end_rapid_quiz_logic(chat_id, context)


async def start_rapid_round(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    state = rapid_games[chat_id]
    difficulty = RAPID_QUIZ_SETTINGS['difficulty']
    
    # Standardized Loading Message
    loading_message = await context.bot.send_message(
        chat_id, 
        rf"🚀 **RAPID QUIZ** \- Round {state['current_round']}/{state['max_rounds']}\n\n"
        rf"**⏳ Downloading and analyzing image\. Please wait\.\.\.**",
        parse_mode='MarkdownV2' 
    )
    
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass
        
    if not pixelated_img_io:
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image acquisition failed\. Rapid Quiz aborted\.", parse_mode='MarkdownV2')
        del rapid_games[chat_id]
        return

    state['answer'] = answer
    state['category'] = category
    state['hint_sentence'] = hint_sentence
    
    if 'timer_task' in state and state['timer_task'] is not None:
        state['timer_task'].cancel()
        state['timer_task'] = None
    
    time_limit = get_rapid_quiz_time_limit(state['current_round'])

    task = asyncio.create_task(timeout_rapid_round_task(chat_id, context, time_limit))
    state['timer_task'] = task
    
    # Standardized Rapid Quiz Caption
    caption = (
        rf"🚀 **RAPID QUIZ** \- Round **{state['current_round']}/{state['max_rounds']}**\n\n"
        rf"**Answer**: **{len(answer)}** letters\. \(Reward: **\+{RAPID_QUIZ_SETTINGS['reward']}** pts\)\n"
        rf"**Time Left**: **{time_limit} seconds**\.\n\n"
        r"Guess the word \*fast\* to continue the streak\! \(No hints\/skips\)"
    )
    
    try:
        sent_message = await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='MarkdownV2' 
        )
        state['last_message_id'] = sent_message.message_id
    except Exception as e:
        logger.error(f"Failed to send rapid quiz photo: {e}")
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image transmission failed\. Rapid Quiz aborted\.", parse_mode='MarkdownV2')
        del rapid_games[chat_id]

async def end_rapid_quiz_logic(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    state = rapid_games.pop(chat_id, None)
    if not state:
        return
        
    user_id = state['user_id']
    total_score = state['total_score']
    
    if 'timer_task' in state and state['timer_task'] is not None:
        state['timer_task'].cancel()
        
    if total_score > 0:
        update_user_score(user_id, total_score)
    
    new_balance = get_user_score(user_id)
    
    rounds_played = state['max_rounds'] if state['current_round'] > state['max_rounds'] else state['current_round'] - 1
    
    # Standardized End Message
    end_message = (
        rf"🏁 **R A P I D   Q U I Z   E N D E D** 🏁\n\n"
        rf"**Total Rounds**: **{rounds_played} / {state['max_rounds']}**\n"
        rf"**Points Earned**: **\+{total_score}**\n"
        rf"**New Balance**: **{new_balance:,}** Points\.\n\n"
        r"Use `/leaderboard` to check your rank\."
    )
    
    await context.bot.send_message(
        chat_id, 
        end_message, 
        parse_mode='MarkdownV2'
    )
    
    await context.bot.send_message(
        chat_id, 
        r"**▶️ S T A R T   N E W   G A M E ?**",
        parse_mode='MarkdownV2',
        reply_markup=get_difficulty_menu() 
    )


async def rapidquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Standardized Pre-check Messages
    if chat_id in active_games:
        await update.message.reply_text(rf"A normal game session is currently active\. Please finish it with `/skip` or submit your guess\.", parse_mode='MarkdownV2')
        return
        
    if chat_id in rapid_games:
        await update.message.reply_text(rf"A Rapid Quiz session is already active\!", parse_mode='MarkdownV2')
        return
        
    rapid_games[chat_id] = {
        'user_id': user_id,
        'total_score': 0,
        'current_round': 1,
        'max_rounds': RAPID_QUIZ_SETTINGS['max_rounds'],
        'timer_task': None,
        'answer': None, 
        'category': None,
        'last_message_id': None
    }
    
    await update.message.reply_text(
        rf"🚀 **R A P I D   Q U I Z** 🚀\n\n"
        r"Starting 10 rounds, with decreasing time limits per image\. Get ready\!", 
        parse_mode='MarkdownV2'
    ) 
    await start_rapid_round(chat_id, context)

async def end_rapid_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(r"Manually ending the Rapid Quiz\.\.\.", parse_mode='MarkdownV2')
    if chat_id not in rapid_games:
        await update.message.reply_text(r"No active Rapid Quiz to end\.", parse_mode='MarkdownV2')
        return
        
    await end_rapid_quiz_logic(chat_id, context)

# ------------------------------------------------------------------
# --- TELEGRAM COMMAND HANDLERS (Normal Mode & Support) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_chat_id(update.effective_chat.id)
    
    # Stylish and Standard Welcome Message
    welcome_text = (
        rf"**👋 Welcome to {BOT_USERNAME}\!** 🖼️\n\n"
        r"I'm a guessing bot that shows you heavily pixelated images\. Your mission is to guess the object or word in the picture\.\n\n" 
        r"**📜 C O M M A N D S**:\n"
        r"• `/game` \- Start a new pixel challenge\.\n"
        r"• `/rapidquiz` \- Start a fast\-paced 10\-round challenge \(Time limit decreases\)\.\n"
        r"• `/myscore` \- Check your points\.\n"
        r"• `/leaderboard` \- See the top players\.\n"
        r"• `/howtoplay` \- Detailed instructions\.\n\n"
        r"Get started with the button below\!"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='start_menu_selector')], 
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    
    try:
        await update.message.reply_photo(
            photo=WELCOME_IMAGE_URL,
            caption=welcome_text,
            parse_mode='MarkdownV2', 
            reply_markup=keyboard
        )
    except Exception:
        await update.message.reply_text(welcome_text, parse_mode='MarkdownV2', reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Stylish and Standard Help Menu
    help_text = (
        r"**📚 B O T   C O M M A N D S**\n\n"
        r"**Game:**\n"
        r"• `/game` \- Start a new pixel challenge \(Normal Mode\)\.\n" 
        r"• `/rapidquiz` \- Start the 10\-round rapid quiz challenge\.\n"
        r"• `/skip` \- Skip the current normal game and reveal the answer\.\n" 
        r"• `/hint` \- Use a hint \(costs points\)\.\n\n" 
        r"**Economy & Stats:**\n"
        r"• `/myscore` \- Check your current point balance\.\n" 
        r"• `/profile` or `/stats` \- View your rank, streak, and album progress\.\n" 
        r"• `/leaderboard` \- See the global top players\.\n" 
        r"• `/daily` \- Claim your daily bonus points\.\n\n" 
        r"**Collection & Shop:**\n"
        r"• `/album` \- View your image categories\.\n" 
        r"• `/shop` \- See and buy special in\-game items\.\n\n" 
        r"Use `/howtoplay` for game details\." 
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='start_menu_selector')], 
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    await update.message.reply_text(help_text, parse_mode='MarkdownV2', reply_markup=keyboard)


async def howtoplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Standardized Rules Text
    rules_text = (
        r"**📜 H O W   T O   P L A Y**\n\n"
        r"**1\. The Goal:** Guess the object, person, or word in the pixelated image\.\n\n"
        r"**2\. Difficulty & Points \(`/game`\):**\n"
        r"• **Easy:** Clearest image, low reward\.\n"
        r"• **Extreme:** Heavily pixelated, high reward\.\n"
        r"The harder the difficulty, the more points you earn\!\n\n"
        r"**3\. Guessing:** Just send the word you think is correct in the chat\. Case and spaces don't matter\.\n\n"
        r"**4\. Hints & Costs:**\n"
        r"• You get one **free hint** \(the category\) with every game\.\n"
        r"• Use `/hint` to reveal a letter in the word \(costs 10 points\)\.\n\n"
        r"**5\. Rapid Quiz \(`/rapidquiz`\):**\n"
        r"• 10 consecutive rounds\.\n"
        r"• **Time decreases** each round \(Starting at 50s, minimum 10s\)\.\n"
        r"• No hints or skips\. Guess correctly to immediately advance\.\n\n"
        r"Good luck, Agent\!" 
    )
    await update.message.reply_text(rules_text, parse_mode='MarkdownV2')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    process = psutil.Process(os.getpid())
    uptime = datetime.now() - start_time
    # Standardized About Text
    about_text = (
        rf"**🤖 A B O U T {BOT_USERNAME}**\n\n"
        r"**Version:** v3\.2 \(Final Flow Fix\)\n"
        r"**Developer:** Ankit / Pixel Team\n"
        r"**Source APIs:** Pexels, Unsplash, Google Gemini AI\n"
        rf"**Uptime:** {escape_markdown_v2(str(timedelta(seconds=int(uptime.total_seconds()))))}\n"
        rf"**Messages Processed:** {total_messages_processed:,}\n\n"
        rf"**Support Group:** [Join here]({SUPPORT_GROUP_LINK})"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Support Group", url=SUPPORT_GROUP_LINK)]
    ])
    await update.message.reply_text(about_text, parse_mode='MarkdownV2', reply_markup=keyboard)


async def photoid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message.photo:
        file_id = update.effective_message.photo[-1].file_id
        await update.message.reply_text(rf"File ID of the photo: `{file_id}`", parse_mode='MarkdownV2')
    elif update.effective_message.reply_to_message and update.effective_message.reply_to_message.photo:
        file_id = update.effective_message.reply_to_message.photo[-1].file_id
        await update.message.reply_text(rf"File ID of the replied photo: `{file_id}`", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text(r"Please send this command as a reply to a photo or as the caption of a photo to get its ID\.", parse_mode='MarkdownV2')


async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    game_state = active_games.pop(chat_id, None)

    if chat_id in rapid_games:
        await update.message.reply_text(r"You cannot skip Rapid Quiz\. Use `/endrapidquiz` to quit\.", parse_mode='MarkdownV2')
        return

    if game_state:
        correct_answer = game_state['answer'].upper()
        letter_count = len(game_state['answer'])
        original_url = game_state['url']
        # Standardized Skip Message
        await update.message.reply_text(
            rf"🛑 **G A M E   S K I P P E D** 🛑\n\nThe correct solution was: **{correct_answer}** \({letter_count} letters\)\.",
            parse_mode='MarkdownV2'
        )
        try:
            await context.bot.send_photo(
                chat_id,
                photo=original_url,
                caption=rf"Original Image\. Solution: **{correct_answer}**\.",
                parse_mode='MarkdownV2'
            )
        except Exception:
            await update.message.reply_text(r"Could not send the original image file\.", parse_mode='MarkdownV2')

        await update.message.reply_text(
            r"**▶️ S T A R T   N E W   G A M E ?**",
            parse_mode='MarkdownV2',
            reply_markup=get_difficulty_menu()
        )
    else:
        await update.message.reply_text(r"No active normal game to skip\. Use `/game` to start one\.", parse_mode='MarkdownV2')


async def simple_hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    game_state = active_games.get(chat_id)
    
    if update.callback_query:
        message_sender = update.callback_query.message
        await update.callback_query.answer()
    else:
        message_sender = update.message
    
    if not game_state:
        await message_sender.reply_text(r"No active normal game to get a hint for\. Use `/game` to start one\.", parse_mode='MarkdownV2')
        return

    if chat_id in rapid_games:
        await message_sender.reply_text(r"Hints are disabled in Rapid Quiz mode\.", parse_mode='MarkdownV2')
        return

    points_cost = GAME_LEVELS[game_state['difficulty']]['hint_cost']
    max_hints = GAME_LEVELS[game_state['difficulty']]['max_hints']
    current_score = get_user_score(user_id)

    if game_state['hints_taken'] >= max_hints:
        await message_sender.reply_text(r"You have already used the maximum number of hints for this game\.", parse_mode='MarkdownV2')
        return

    if current_score < points_cost:
        # Standardized Insufficient Funds Message
        await message_sender.reply_text(
            rf"❌ **I N S U F F I C I E N T   F U N D S** ❌\n\n"
            rf"You need **{points_cost}** Points for a hint\. Current balance: **{current_score}**\.", 
            parse_mode='MarkdownV2'
        )
        return

    # Deduct cost and save hint usage
    update_user_score(user_id, -points_cost)
    game_state['hints_taken'] += 1
    
    answer = game_state['answer']
    current_hint_list = list(game_state['current_hint_string'])
    
    unrevealed_indices = [i for i, char in enumerate(current_hint_list) if char == '_']
    
    if not unrevealed_indices:
        # This shouldn't happen if max_hints check is correct, but safe fallback
        await message_sender.reply_text(r"All letters have already been revealed\!", parse_mode='MarkdownV2')
        return

    # Reveal a random unrevealed letter
    index_to_reveal = random.choice(unrevealed_indices)
    current_hint_list[index_to_reveal] = answer[index_to_reveal].upper()
    
    new_hint_string = "".join(current_hint_list)
    game_state['current_hint_string'] = new_hint_string
    
    new_balance = get_user_score(user_id)
    
    # Standardized Hint Message
    hint_message = (
        rf"💡 **H I N T   R E V E A L E D** 💡\n\n"
        rf"**Cost**: **\-{points_cost}** Points\.\n"
        rf"**Progress**: `{new_hint_string}`\n"
        rf"**Hints Left**: **{max_hints - game_state['hints_taken']}**\n\n"
        rf"**New Balance**: **{new_balance:,}** Points\."
    )

    try:
        # Edit the previous game message caption (if it exists)
        await context.bot.edit_message_caption(
            chat_id=chat_id, 
            message_id=game_state['message_id'],
            caption=game_state['current_caption'].replace(game_state['old_hint_string'], new_hint_string),
            parse_mode='MarkdownV2',
            reply_markup=message_sender.reply_markup # Keep buttons if any
        )
    except Exception:
        # Send a new message if editing fails
        pass 
    
    game_state['old_hint_string'] = new_hint_string
    await message_sender.reply_text(hint_message, parse_mode='MarkdownV2')
    

async def myscore_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    score = get_user_score(user_id)
    # Standardized Score Message
    await update.message.reply_text(
        rf"💰 **C U R R E N T   B A L A N C E** 💰\n\n"
        rf"Your current score is **{score:,}** Points\.\n\n"
        r"Use `/leaderboard` to see the top players\.",
        parse_mode='MarkdownV2'
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    rank, score, solved_count, current_streak, _ = await get_user_profile_data(user_id)
    
    user_name = escape_markdown_v2(update.effective_user.first_name)
    
    # Standardized Profile Text
    profile_text = (
        rf"👤 **A G E N T   P R O F I L E** 📊\n\n"
        rf"**Agent Name**: {user_name}\n"
        rf"**Global Rank**: **\#{rank}**\n"
        rf"**Total Score**: **{score:,}** Points\n\n"
        rf"**Solved Challenges**: **{solved_count}**\n"
        rf"**Current Streak**: **{current_streak}** 🔥\n\n"
        r"Keep solving to climb the leaderboard\!"
    )

    await update.message.reply_text(profile_text, parse_mode='MarkdownV2')


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_time_utc = datetime.now(pytz.utc)
    
    _, _, _, _, last_claim = await get_user_profile_data(user_id)
    
    # Check if user has claimed within the last 24 hours
    if last_claim and (current_time_utc - last_claim) < timedelta(hours=23, minutes=59):
        next_claim_time = last_claim + timedelta(days=1)
        time_left = next_claim_time - current_time_utc
        hours, remainder = divmod(time_left.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        
        # Standardized Cooldown Message
        cooldown_text = (
            rf"⏳ **C L A I M   O N   C O O L D O W N** ⏳\n\n"
            rf"You can claim your daily bonus again in **{int(hours)} hours and {int(minutes)} minutes**\.\n\n"
            r"Keep playing with `/game` to earn more points now\!"
        )
        await update.message.reply_text(cooldown_text, parse_mode='MarkdownV2')
        return

    # Claim the bonus
    success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
    new_balance = get_user_score(user_id)
    
    if success:
        # Standardized Success Message
        success_text = (
            rf"✅ **D A I L Y   R E W A R D   C L A I M E D** ✅\n\n"
            rf"You received **\+{DAILY_BONUS_POINTS}** Points\.\n"
            rf"**New Balance**: **{new_balance:,}** Points\.\n\n"
            r"Come back in 24 hours to claim your next bonus\!"
        )
        await update.message.reply_text(success_text, parse_mode='MarkdownV2')
    else:
        await update.message.reply_text(r"❌ **Error**\: Could not process daily claim\. Please try again later\.", parse_mode='MarkdownV2')


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top_scores = get_top_scores(limit=10)
    
    if not top_scores:
        await update.message.reply_text(r"📊 **L E A D E R B O A R D**\n\nNo scores recorded yet\.", parse_mode='MarkdownV2')
        return
        
    leaderboard_entries = []
    
    # Fetch user names
    for rank, (user_id, score) in enumerate(top_scores):
        user = await context.bot.get_chat(user_id)
        user_name = escape_markdown_v2(user.first_name)
        
        icon = ""
        if rank == 0:
            icon = "🥇"
        elif rank == 1:
            icon = "🥈"
        elif rank == 2:
            icon = "🥉"
        
        leaderboard_entries.append(
            rf"{icon} \#{rank + 1}\. {user_name} \- **{score:,}** Points"
        )
    
    leaderboard_text = (
        r"🏆 **T O P   P L A Y E R S** 🏆\n\n"
        + "\n".join(leaderboard_entries)
    )
    
    await update.message.reply_text(leaderboard_text, parse_mode='MarkdownV2')


async def album_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    collection = await get_user_collection(user_id)
    
    total_categories = len(SEARCH_CATEGORIES)
    
    album_text = (
        rf"🖼️ **P I X E L M A S T E R   A L B U M** 🖼️\n\n"
        rf"**Total Categories Collected**: **{len(collection)} / {total_categories}**\n\n"
    )
    
    # Organize collected categories
    collected_icons = []
    uncollected_count = 0
    for category, emoji in SEARCH_CATEGORIES.items():
        if category in collection:
            collected_icons.append(emoji)
        else:
            uncollected_count += 1
            
    collected_icons.sort()
    
    if collected_icons:
        album_text += r"**Collected Categories:**\n"
        # Join collected emojis in chunks for better display
        chunk_size = 10 
        for i in range(0, len(collected_icons), chunk_size):
            album_text += "".join(collected_icons[i:i + chunk_size]) + "\n"
        album_text += "\n"
    
    album_text += rf"**Uncollected Categories Left**: **{uncollected_count}**\n\n"
    album_text += r"Solve more games to complete your album\!\n"
    
    # Add a button to view specific categories (if needed, placeholder for future)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Back to Menu", callback_data='help_menu')]
    ])
    
    await update.message.reply_text(album_text, parse_mode='MarkdownV2', reply_markup=keyboard)


async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    user_id = query.from_user.id
    
    item_id = query.data.split('_')[1]
    item = SHOP_ITEMS.get(item_id)
    
    if not item:
        await context.bot.send_message(chat_id, r"❌ **Error**: Invalid shop item selected\.", parse_mode='MarkdownV2')
        return
        
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    if current_score < cost:
        await context.bot.send_message(
            chat_id, 
            rf"❌ **I N S U F F I C I E N T   F U N D S** ❌\n\n"
            rf"You need **{cost}** Points to buy **{item['name']}**\.\n"
            rf"Current balance: **{current_score}**\.",
            parse_mode='MarkdownV2'
        )
        return
        
    # Implement actual item purchase logic here (e.g., granting user item count in DB)
    # For now, just deduct points
    update_user_score(user_id, -cost)
    new_balance = get_user_score(user_id)
    
    # Standardized Purchase Success Message
    success_message = (
        rf"🛒 **P U R C H A S E   S U C C E S S F U L** 🛒\n\n"
        rf"You successfully bought **{item['name']}**\.\n"
        rf"**Cost**: **\-{cost}** Points\.\n"
        rf"**New Balance**: **{new_balance:,}** Points\.\n\n"
        r"Item effects will be active in your next game\!"
    )
    
    await context.bot.send_message(chat_id, success_message, parse_mode='MarkdownV2')


async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Standardized Shop Menu
    shop_text = r"🛍️ **S H O P   M E N U** 🛍️\n\n"
    
    keyboard = []
    for item_id, item in SHOP_ITEMS.items():
        shop_text += rf"**{item['name']}** \({item['cost']} Points\)\n"
        shop_text += rf"• {item['description']}\n\n"
        keyboard.append([InlineKeyboardButton(f"Buy {item['name']} ({item['cost']} Points)", callback_data=f'buy_{item_id}')])
        
    await update.message.reply_text(
        shop_text + r"Select an item to purchase\:", 
        parse_mode='MarkdownV2',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# ------------------------------------------------------------------
# --- GAME LOGIC HANDLERS (Normal Mode) ---
# ------------------------------------------------------------------

async def fetch_and_start_game(chat_id: int, context: ContextTypes.DEFAULT_TYPE, difficulty: str):
    
    # Clear any existing game in this chat
    if chat_id in rapid_games:
        # Cancel running rapid quiz timer if any
        if 'timer_task' in rapid_games[chat_id] and rapid_games[chat_id]['timer_task'] is not None:
            rapid_games[chat_id]['timer_task'].cancel()
        del rapid_games[chat_id]
        
    if chat_id in active_games:
        active_games.pop(chat_id)
        
    # Standardized Loading Message
    loading_message = await context.bot.send_message(
        chat_id, 
        rf"**🖼️ Starting New Game \- {difficulty.upper()}**\n\n"
        rf"**⏳ Downloading and analyzing image\. Please wait\.\.\.**",
        parse_mode='MarkdownV2' 
    )

    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass
        
    if not pixelated_img_io:
        # Standardized Error Message
        await context.bot.send_message(
            chat_id, 
            rf"❌ **Game Creation Failed** ❌\n\n"
            rf"Error: {escape_markdown_v2(answer)}\n\n"
            r"Please try again later\.", 
            parse_mode='MarkdownV2'
        )
        return
    
    points = GAME_LEVELS[difficulty]['points']
    
    # Initialize hint string (e.g., '____')
    initial_hint_string = '_ ' * len(answer)
    initial_hint_string = initial_hint_string[:-1]
    
    category_emoji = SEARCH_CATEGORIES.get(category, '❓')
    
    # Standardized Game Caption (using raw f-string for better variable injection and escaping)
    caption = (
        rf"**📸 V I S U A L   C H A L L E N G E: {difficulty.upper()}**\n\n"
        r"Identify the object in this high\-pixel density image\.\n\n"
        rf"**🎁 Reward**: **\+{points} Points**\n"
        rf"**🧩 Progress**: `{initial_hint_string}` \({len(answer)} letters\)\n"
        rf"**💡 Free Clue**: \*Category is {category_emoji} {category.capitalize()}\*\n\n"
        r"Use `/hint` to reveal a letter \(Costs 10 Points\)"
    )
    
    
    # Save game state
    active_games[chat_id] = {
        'answer': answer,
        'difficulty': difficulty,
        'points': points,
        'url': original_url,
        'category': category,
        'hint_sentence': hint_sentence,
        'hints_taken': 0,
        'current_hint_string': initial_hint_string,
        'old_hint_string': initial_hint_string, # Store to easily replace next time
        'message_id': None, # Will be filled below
        'current_caption': caption
    }
    
    try:
        sent_message = await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='MarkdownV2' 
        )
        active_games[chat_id]['message_id'] = sent_message.message_id
    except Exception as e:
        logger.error(f"Failed to send game photo: {e}")
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image transmission failed\. Game aborted\.", parse_mode='MarkdownV2')
        del active_games[chat_id]


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text(r"A game is already active\. Guess the word or use `/skip` to start a new one\.", parse_mode='MarkdownV2')
        return

    # Send the difficulty menu
    await send_difficulty_menu(update, context)


async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    data = query.data
    
    if data == 'start_menu_selector':
        await send_difficulty_menu(update, context)
        return

    if data.startswith('game_'):
        difficulty = data.split('_')[1]
        
        # Check if a game is already running (double check for safety)
        if chat_id in active_games or chat_id in rapid_games:
            # Edit the message to reflect status and suggest next steps
            text = r"A game is already active or starting in this chat\. Please wait or use `/skip`\."
            await query.edit_message_text(text, parse_mode='MarkdownV2', reply_markup=None)
            return

        # Replace the difficulty menu with a confirmation/start message
        start_message = rf"Starting a **{difficulty.upper()}** challenge\.\.\."
        await query.edit_message_text(start_message, parse_mode='MarkdownV2', reply_markup=None)
        
        # Start the actual game fetch task
        await fetch_and_start_game(chat_id, context, difficulty)
        
    elif data == 'game_end':
        # Handles user pressing 'End Game' (if implemented later)
        if chat_id in active_games:
            active_games.pop(chat_id)
            await query.edit_message_text(r"Game session manually terminated\.", parse_mode='MarkdownV2', reply_markup=None)
        else:
            await query.edit_message_text(r"No active game to terminate\.", parse_mode='MarkdownV2', reply_markup=None)


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, guess: str):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = escape_markdown_v2(update.effective_user.first_name)
    
    # -------------------
    # --- RAPID QUIZ CHECK ---
    # -------------------
    if chat_id in rapid_games:
        state = rapid_games[chat_id]
        correct_answer = state['answer']
        reward = RAPID_QUIZ_SETTINGS['reward']
        
        if guess == correct_answer:
            # Correct Guess in Rapid Quiz
            
            # Cancel the timeout timer
            if state['timer_task'] is not None:
                state['timer_task'].cancel()
                state['timer_task'] = None
                
            state['total_score'] += reward
            
            # Standardized Rapid Quiz Success Message (edit photo caption)
            caption = (
                rf"✅ **S O L V E D \!** ✅\n\n"
                rf"**Agent {user_name}** solved Round **{state['current_round']}** with: **{correct_answer.upper()}**\n\n"
                rf"**💰 Reward**: **\+{reward} Points**\n"
                rf"**Total Quiz Score**: **{state['total_score']}**"
            )

            try:
                # Edit the previous image caption
                await context.bot.edit_message_caption(
                    chat_id=chat_id,
                    message_id=state['last_message_id'],
                    caption=caption,
                    parse_mode='MarkdownV2',
                    reply_markup=None
                )
            except Exception:
                 await context.bot.send_message(
                    chat_id,
                    caption,
                    parse_mode='MarkdownV2'
                )
            
            # Move to next round
            state['current_round'] += 1
            
            if state['current_round'] > state['max_rounds']:
                await end_rapid_quiz_logic(chat_id, context)
            else:
                # Give a short delay before next round starts
                await context.bot.send_message(chat_id, r"Starting next round in 3 seconds\.\.\.", parse_mode='MarkdownV2')
                await asyncio.sleep(3)
                await start_rapid_round(chat_id, context)

            return True

        else:
            # Wrong Guess in Rapid Quiz (No hint, just notify)
            await update.message.reply_text(r"❌ **I N C O R R E C T** ❌\. Try again quickly\!", parse_mode='MarkdownV2')
            return False

    # -------------------
    # --- NORMAL GAME CHECK ---
    # -------------------
    game_state = active_games.get(chat_id)
    
    if not game_state:
        # If no game is active, ignore the message or send a polite prompt
        if update.message.text and len(update.message.text.split()) == 1 and update.message.text.isalpha():
            # Looks like an accidental guess
            await update.message.reply_text(r"No active game running\. Use `/game` to start a new pixel challenge\.", parse_mode='MarkdownV2')
        return False
        
    correct_answer = game_state['answer']
    
    if guess == correct_answer:
        # Correct Guess in Normal Game
        active_games.pop(chat_id) # End the game
        
        points = game_state['points']
        category = game_state['category']
        original_url = game_state['url']
        letter_count = len(correct_answer)
        
        update_user_score(user_id, points)
        await save_solved_image(user_id, category) # Save to collection and update streak/count
        
        current_score = get_user_score(user_id)
        
        # Standardized Normal Game Success Message
        caption = (
            rf"✅ **S O L U T I O N   A C Q U I R E D** ✅\n\n"
            rf"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** \({letter_count} letters\)\n\n"
            rf"**💰 Reward**: **\+{points} Points**\n"
            rf"**Current Balance**: **{current_score:,}** Points\n\n"
            r"View the original image below\."
        )
        
        # Edit the original message caption to show solved status
        try:
            await context.bot.edit_message_caption(
                chat_id=chat_id,
                message_id=game_state['message_id'],
                caption=caption,
                parse_mode='MarkdownV2',
                reply_markup=None
            )
        except Exception:
            # If editing fails (e.g., old message), send a new text message
            await update.message.reply_text(caption, parse_mode='MarkdownV2')

        # Send the original image
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=rf"Original Image\. Solution: **{correct_answer.upper()}**\.",
                parse_mode='MarkdownV2'
            )
        except Exception:
            await update.message.reply_text(r"Could not send the original image file\.", parse_mode='MarkdownV2')
            
        # Prompt for next game
        await update.message.reply_text(
            r"**▶️ S T A R T   N E W   G A M E ?**",
            parse_mode='MarkdownV2',
            reply_markup=get_difficulty_menu()
        )
        
        return True
    else:
        # Wrong Guess in Normal Game
        letter_count = len(correct_answer)
        # Standardized Wrong Guess Message
        await update.message.reply_text(
            rf"❌ **I N C O R R E C T** ❌\n\n"
            rf"Try again, Agent\. The word is **{letter_count}** letters long\. Use `/hint` if you need help\.",
            parse_mode='MarkdownV2'
        )
        return False


async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global total_messages_processed
    total_messages_processed += 1
    
    if update.effective_message.text:
        text = update.effective_message.text
        
        # Check if the message is a potential guess (must contain only letters and be one word)
        # Or if it's a message in a rapid quiz (where any text is a guess)
        is_potential_guess = False
        if chat_id in rapid_games:
            is_potential_guess = True
        elif len(text.split()) == 1 and text.isalpha() and len(text) <= 15:
            is_potential_guess = True

        if is_potential_guess:
            # Clean guess: remove spaces, convert to lowercase
            cleaned_guess = re.sub(r'[^a-z]', '', text.lower())
            
            if cleaned_guess:
                await check_guess_and_update_score(update, context, cleaned_guess)
                return 

    elif update.effective_message.caption:
        # Handle guesses made via captions on photos/files
        caption = update.effective_message.caption
        if len(caption.split()) == 1 and caption.isalpha() and len(caption) <= 15:
            cleaned_guess = re.sub(r'[^a-z]', '', caption.lower())
            
            if cleaned_guess:
                await check_guess_and_update_score(update, context, cleaned_guess)
                return 

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Standardized Voice Message Reply
    await update.message.reply_text(r"Sorry, I cannot process voice messages for game guesses\. Please type your answer\.", parse_mode='MarkdownV2')


async def handle_core_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == 'help_menu':
        await help_command(query, context)
    elif data.startswith('album_view_'):
        # Placeholder for future detailed album view
        await context.bot.send_message(query.message.chat_id, r"Album details coming soon\!", parse_mode='MarkdownV2')


# ------------------------------------------------------------------
# --- ADMIN & MAINTENANCE COMMANDS ---
# ------------------------------------------------------------------

def is_owner(user_id: int):
    return user_id == BROADCAST_ADMIN_ID

def check_owner_wrapper(func):
    """Decorator to restrict command access to the bot owner."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if is_owner(update.effective_user.id):
            return await func(update, context)
        else:
            await update.message.reply_text(r"❌ **Access Denied**\: This command is restricted to the bot owner\.", parse_mode='MarkdownV2')
            return
    return wrapper

@check_owner_wrapper
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        # Standardized Admin Message
        await update.message.reply_text(r"**Usage**: `/broadcast <message>`", parse_mode='MarkdownV2')
        return
        
    broadcast_message = " ".join(context.args)
    conn, error = db_connect()
    if error:
        await update.message.reply_text(rf"❌ **DB Error**: {escape_markdown_v2(error)}", parse_mode='MarkdownV2')
        return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM known_chats;")
            all_chats = [row[0] for row in cur.fetchall()]
    except Exception as e:
        await update.message.reply_text(rf"❌ **DB Fetch Error**: {escape_markdown_v2(str(e))}", parse_mode='MarkdownV2')
        return
        
    success_count = 0
    failure_count = 0
    
    # Standardized Broadcast Header
    final_message = r"📢 **B R O A D C A S T   M E S S A G E** 📢\n\n" + broadcast_message
    
    for chat_id in all_chats:
        try:
            await context.bot.send_message(chat_id, final_message, parse_mode='MarkdownV2')
            success_count += 1
            await asyncio.sleep(0.05) # Small delay to avoid flooding limits
        except Exception:
            failure_count += 1
            
    await update.message.reply_text(
        rf"✅ **Broadcast Complete**\: Sent to **{success_count}** chats\. Failed for **{failure_count}** chats\.",
        parse_mode='MarkdownV2'
    )

async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    new_members = update.message.new_chat_members
    
    for member in new_members:
        # Ignore bot itself
        if member.is_bot:
            continue
            
        user_name = escape_markdown_v2(member.first_name)
        
        # Standardized Welcome Message for Group
        welcome_text = (
            rf"👋 **W E L C O M E** \!\n\n"
            rf"Hey, {user_name} \! Welcome to the group\.\n"
            r"I am **Pixel Peep**\. You can play the image guessing game with me\.\n\n"
            r"Use `/start` to see my commands\."
        )
        
        try:
            await update.message.reply_text(welcome_text, parse_mode='MarkdownV2')
        except Exception:
            pass 

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    
    if update and hasattr(update, 'effective_chat'):
        chat_id = update.effective_chat.id
        # Standardized Error Notification
        error_msg = r"⚠️ **S Y S T E M   E R R O R** ⚠️\n\n"
        error_msg += r"An unexpected error occurred\. We've been notified and are investigating\.\n"
        error_msg += r"Please try again or use `/help`\."
        
        try:
            # Try to send a polite error message to the user/chat
            await context.bot.send_message(chat_id, error_msg, parse_mode='MarkdownV2')
        except Exception:
            pass


# ------------------------------------------------------------------
# --- MAIN RUN FUNCTION ---
# ------------------------------------------------------------------

def main():
    db_connect() 
    load_known_users()
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Public Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("game", game_command))
    application.add_handler(CommandHandler("rapidquiz", rapidquiz_command))
    application.add_handler(CommandHandler("myscore", myscore_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("stats", profile_command)) # Alias
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("hint", simple_hint_command))
    application.add_handler(CommandHandler("howtoplay", howtoplay_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("album", album_command))
    application.add_handler(CommandHandler("shop", shop_command))
    
    # Utility Command
    application.add_handler(CommandHandler("photoid", photoid_command))

    # Owner-Only/Utility Commands
    application.add_handler(CommandHandler("broadcast", check_owner_wrapper(broadcast_command))) 
    application.add_handler(CommandHandler("endrapidquiz", end_rapid_quiz)) 

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION), 
        process_message
    ))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Callback Query Handler 
    application.add_handler(CallbackQueryHandler(handle_game_callback, pattern=r'^(game_|game_end|start_menu_selector)')) 
    application.add_handler(CallbackQueryHandler(handle_core_callback, pattern=r'^(help_menu|album_view_)'))
    application.add_handler(CallbackQueryHandler(buy_command, pattern=r'^buy_')) 

    application.add_error_handler(error_handler)
    
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
        logger.info("Bot started with polling.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
