# main.py (Final, Complete, Speed-Optimized, with Segmentation and Expanded Categories)

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
    # This warning is harmless if the user has google-genai installed later
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Fetch Gemini Key

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

# Store active games in memory (Normal Mode - Single Round)
# Structure: {chat_id: {'answer': 'keyword', 'difficulty': 'easy', 'url': 'original_url', 'hints_taken': 0, 'hint_string': '_______', 'category': 'animal', 'hint_sentence': '...'}}
active_games = defaultdict(dict) 

# Store fast-mode games (Multi-Round, Timed)
# Structure: {chat_id: {'user_id': id, 'score': 0, 'current_round': 1, 'max_rounds': 10, 'timer_task': asyncio.Task, 'total_score': 0, 'answer': '...', 'category': '...'}}
fast_games = defaultdict(dict) 

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
# NOTE: Replace with your actual photo ID if using /photoid command to get it.
WELCOME_IMAGE_URL = "AgACAgUAAxkBAAPNaSdU6DYWrJbMvkAsoRF93H3V2x8AAhYLaxsfjEFVReKLNrpOrBUBAAMCAAN4AAM2BA" 
# NOTE: Replace with your actual QR code photo ID
DONATION_QR_CODE_ID = "AgACAgUAAxkBAAIBzGkn8CZvfaDPAckxv-cOPFgKlus4AAJdC2sbmhxBVWBRzMvm7w0HAQADAgADeQADNgQ" 
#UPI_ID_FOR_DONATION = "your_upi_id@bank"

# ADJUSTED PIXELATION LEVELS: downscale_factor is the divisor for image dimensions.
GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 3, 'hint_cost': 10}, 
    'hard': {'downscale_factor': 15, 'points': 70, 'max_hints': 3, 'hint_cost': 10}, 
    'medium': {'downscale_factor': 10, 'points': 50, 'max_hints': 3, 'hint_cost': 10}, 
    'easy': {'downscale_factor': 25, 'points': 30, 'max_hints': 3, 'hint_cost': 10} # Easiest mode now has highest factor for clearest image
}

# EXPANDED SEARCH CATEGORIES for better variety (SIMULATING 1000 items)
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

# ------------------------------------------------------------------
# --- DATABASE CONNECTION AND DATA MANAGEMENT FUNCTIONS ---
# ------------------------------------------------------------------

def db_connect():
    """Establishes and returns a PostgreSQL connection, and creates/updates necessary tables."""
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

            # Add missing columns if they don't exist (handle legacy tables)
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
                INSERT INTO game_scores (user_id, score, solved_count, current_streak) 
                VALUES (%s, %s, 0, 0)
                ON CONFLICT (user_id) 
                DO UPDATE SET score = game_scores.score + EXCLUDED.score;
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

async def get_user_profile_data(user_id: int):
    """Retrieves all profile data for a user, including rank."""
    conn, error = db_connect()
    if error: return None, 0, 0, None, "DB Error"
    
    try:
        with conn.cursor() as cur:
            # Get user stats
            cur.execute(
                "SELECT score, solved_count, current_streak, last_daily_claim FROM game_scores WHERE user_id = %s;",
                (user_id,)
            )
            data = cur.fetchone()
            if not data:
                # Initialize user if not found
                update_user_score(user_id, 0)
                # Set a past date for first claim availability
                data = (0, 0, 0, datetime.now(pytz.utc) - timedelta(days=2))
            
            score, solved_count, current_streak, last_daily_claim = data

            # Calculate Rank
            cur.execute("SELECT COUNT(*) FROM game_scores WHERE score > %s;", (score,))
            rank = cur.fetchone()[0] + 1
            
            return rank, score, solved_count, current_streak, last_daily_claim
    except Exception as e:
        logger.error(f"Error fetching profile data for user {user_id}: {e}")
        return None, 0, 0, 0, f"Error: {e}"

async def update_daily_claim(user_id: int, points: int):
    """Updates last_daily_claim timestamp and adds points."""
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
    """Retrieves all collected categories for a user."""
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
    """Updates solved count, streak, and collection after a correct guess."""
    conn, error = db_connect()
    if error: return
    try:
        with conn.cursor() as cur:
            # 1. Update Collection (sticker album)
            # We use the original search category for the album
            cur.execute(
                "INSERT INTO user_collection (user_id, category) VALUES (%s, %s) ON CONFLICT (user_id, category) DO NOTHING;",
                (user_id, category)
            )

            # 2. Update solved_count and current_streak
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
# --- GEMINI AI INTEGRATION FUNCTIONS ---
# ------------------------------------------------------------------

def initialize_gemini_client():
    """Initializes and returns the Gemini client."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. AI functions will fail.")
        return None
    try:
        # The client automatically uses the key from the environment
        client = google.genai.Client()
        return client
    except Exception as e:
        logger.error(f"Gemini Client Initialization Error: {e}")
        return None

async def get_ai_answer_and_hint(image_data: bytes) -> tuple[str | None, str | None]:
    """
    Sends image data to Gemini 2.5 Flash to get a one-word answer and a hint sentence.
    Returns (answer, hint_sentence).
    """
    client = initialize_gemini_client()
    if client is None:
        return None, None
        
    image_part = types.Part.from_bytes(
        data=image_data,
        mime_type='image/jpeg'
    )

    # Prompt engineered to force a JSON output for reliability
    prompt = (
        "Analyze this photograph. Your task is to provide: "
        "1. A single, specific, common noun or object name as the **answer**. The word should contain only letters (no spaces or hyphens) and be 3 to 15 characters long. "
        "2. A single, short, descriptive sentence as a **hint** for that answer. "
        "Strictly return ONLY a JSON object with two keys: 'answer' (one word, lowercase) and 'hint' (one sentence)."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[image_part, prompt]
        )
        
        # Parse the AI's JSON output
        json_text = response.text.strip().replace("```json", "").replace("```", "").replace("`", "").strip()
        result = json.loads(json_text)
        
        # Clean and validate the answer
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
    """Fetches image data from Pexels API."""
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
    """Fetches image data from Unsplash API."""
    if not UNSPLASH_ACCESS_KEY:
        raise Exception("UNSPLASH_ACCESS_KEY is missing.")
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    url = f"https://api.unsplash.com/photos/random?query={urllib.parse.quote(query)}&orientation=squarish" 
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photo = response.json() 
    image_url = photo['urls']['regular']
    return image_url, query, "Unsplash"

# Main Image Processing Function (SPEED OPTIMIZED)
async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None, str | None, str | None]:
    """Randomly selects an API, fetches, sends a compressed image to AI, and pixelates. Returns (io, AI_answer, url, category, AI_hint_sentence)."""
    category = random.choice(search_queries) # Original search category
    query = category
    
    available_apis = []
    if PEXELS_API_KEY:
        available_apis.append(fetch_image_from_pexels)
    if UNSPLASH_ACCESS_KEY:
        available_apis.append(fetch_image_from_unsplash)

    if not available_apis:
        return None, "Both API Keys Missing. Check .env file.", None, None, None

    fetcher = random.choice(available_apis)
    
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
        # 1. Download Image Content
        image_response = requests.get(image_url, stream=True, timeout=10)
        image_response.raise_for_status()
        image_data = image_response.content
        
        # 2. Get AI Answer and Hint (Optimization: Use a smaller image for AI)
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
        
        # 3. Process/Pixelate Image using PIL
        
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
        
        # Prepare output
        output = io.BytesIO()
        pixelated_img.save(output, format='JPEG')
        output.seek(0)
        
        # Return the AI-generated answer and hint
        return output, ai_answer, image_url, category, hint_sentence
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image Download Error: {e}")
        return None, f"Image Download Error: {e}", None, None, None
    except Exception as e:
        logger.error(f"Image Processing/Unknown Error: {e}")
        return None, f"Image Processing/Unknown Error: {e}", None, None, None


# ------------------------------------------------------------------
# --- FAST MODE GAME FUNCTIONS (New/Updated) ---
# ------------------------------------------------------------------

FAST_MODE_SETTINGS = {
    'difficulty': 'medium', # Fixed difficulty for fast mode
    'reward': 20,           # Fixed points per correct answer
    'max_rounds': 10,
    'time_limit': 10        # Seconds per image (as requested)
}

async def timeout_fast_round(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Automatically advances the fast game after the time limit."""
    
    if chat_id not in fast_games:
        return
        
    state = fast_games[chat_id]
    
    # Try to clean up the last sent message (the pixelated image)
    try:
        if 'last_message_id' in state:
            await context.bot.edit_message_caption(
                chat_id=chat_id, 
                message_id=state['last_message_id'],
                caption=f"⏱️ **TIME UP!** The answer was **{state['answer'].upper()}**.",
                reply_markup=None,
                parse_mode='Markdown'
            )
    except Exception:
        # Ignore if the message was already edited or deleted
        pass 

    await context.bot.send_message(
        chat_id=chat_id, 
        text=f"⏱️ **Round {state['current_round']} Timeout!** The answer was **{state['answer'].upper()}**."
    )
    
    state['current_round'] += 1
    
    if state['current_round'] > state['max_rounds']:
        await end_fast_game(chat_id, context)
    else:
        await start_fast_round(chat_id, context)

async def start_fast_round(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Fetches, pixelates, and sends the next fast game round image."""
    state = fast_games[chat_id]
    difficulty = FAST_MODE_SETTINGS['difficulty']
    
    # 1. Send loading message
    loading_message = await context.bot.send_message(
        chat_id, 
        f"🚀 **Fast Round {state['current_round']}/{state['max_rounds']}**: Downloading and analyzing image..."
    )
    
    # 2. Fetch image and AI data (simultaneously)
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    if not pixelated_img_io:
        await context.bot.send_message(chat_id, f"❌ **Error**: Image acquisition failed. Fast Game aborted.")
        del fast_games[chat_id]
        return

    # Update state with new game data
    state['answer'] = answer
    state['category'] = category
    state['hint_sentence'] = hint_sentence
    
    # 3. Cancel any previous timer
    if 'timer_task' in state and state['timer_task'] and not state['timer_task'].done():
        state['timer_task'].cancel()

    # 4. Start new timer task
    time_limit = FAST_MODE_SETTINGS['time_limit']
    # Create the task that will call timeout_fast_round after the time limit
    state['timer_task'] = asyncio.create_task(asyncio.sleep(time_limit))
    state['timer_task'].add_done_callback(
        lambda t: asyncio.create_task(timeout_fast_round(chat_id, context)) if not t.cancelled() else None
    )
    
    # 5. Send image
    caption = (
        f"🚀 **FAST MODE: Round {state['current_round']}/{state['max_rounds']}**\n\n"
        f"**Answer**: **{len(answer)}** letters. (Reward: **+{FAST_MODE_SETTINGS['reward']}** pts)\n"
        f"**Time Left**: **{time_limit} seconds**.\n"
        f"Guess the word *fast* to continue the streak! (No hints/skips)"
    )
    
    try:
        # Delete the loading message, send the image
        await context.bot.delete_message(chat_id, loading_message.message_id)
        
        sent_message = await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='Markdown'
        )
        # Store message ID to edit on timeout/correct guess
        state['last_message_id'] = sent_message.message_id
    except Exception as e:
        logger.error(f"Failed to send fast game photo: {e}")
        await context.bot.send_message(chat_id, "❌ **Error**: Image transmission failed. Fast Game aborted.")
        del fast_games[chat_id]

async def end_fast_game(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Finalizes the fast game, updates user score, and cleans up."""
    state = fast_games.pop(chat_id, None)
    if not state:
        return
        
    user_id = state['user_id']
    total_score = state['total_score']
    
    # Cancel any running timer task
    if 'timer_task' in state and state['timer_task'] and not state['timer_task'].done():
        state['timer_task'].cancel()
        
    if total_score > 0:
        update_user_score(user_id, total_score)
    
    new_balance = get_user_score(user_id)
    
    # current_round is 1 more than the number of completed rounds
    rounds_played = state['max_rounds'] if state['current_round'] > state['max_rounds'] else state['current_round'] - 1
    
    end_message = (
        f"🏁 **F A S T   M O D E   E N D E D** 🏁\n\n"
        f"**Total Correct**: **{rounds_played} / {state['max_rounds']}**\n"
        f"**Points Earned**: **+{total_score}**\n"
        f"**New Balance**: **{new_balance:,}** Points.\n\n"
    )
    
    # Add /game button (as requested)
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start New Game", callback_data='game_easy')]])
    
    await context.bot.send_message(chat_id, end_message, parse_mode='Markdown', reply_markup=keyboard)


async def fastgame_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts a new fast-paced 10-round game."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    if chat_id in active_games:
        await update.message.reply_text("A normal game session is currently active. Please finish it with `/skip` or submit your guess.")
        return
        
    if chat_id in fast_games:
        await update.message.reply_text("A fast game session is already active!")
        return
        
    # Initialize fast game state
    fast_games[chat_id] = {
        'user_id': user_id,
        'total_score': 0,
        'current_round': 1,
        'max_rounds': FAST_MODE_SETTINGS['max_rounds'],
        'timer_task': None,
        'answer': None, 
        'category': None,
        'last_message_id': None
    }
    
    await update.message.reply_text("🚀 **FAST MODE** started! Get ready for 10 rounds, 10 seconds each!", parse_mode='Markdown')
    await start_fast_round(chat_id, context)

# ------------------------------------------------------------------
# --- TELEGRAM COMMAND HANDLERS (Normal Mode & Support) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message with an image and saves chat ID."""
    save_chat_id(update.effective_chat.id)
    
    welcome_text = (
        f"**Welcome to {BOT_USERNAME}!** 🖼️\n\n"
        f"I'm a guessing bot that shows you heavily pixelated images. Your mission is to guess the object or word in the picture.\n\n"
        f"**Commands:**\n"
        f"• `/game` - Start a new pixel challenge.\n"
        f"• `/fastgame` - Start a fast-paced 10-round challenge (10s limit).\n"
        f"• `/myscore` - Check your points.\n"
        f"• `/leaderboard` - See the top players.\n"
        f"• `/howtoplay` - Detailed instructions.\n\n"
        f"Get started with `/game`!"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='game_easy')],
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    
    try:
        # Send a photo with the welcome message
        await update.message.reply_photo(
            photo=WELCOME_IMAGE_URL,
            caption=welcome_text,
            parse_mode='Markdown',
            reply_markup=keyboard
        )
    except Exception:
        # Fallback to text message if photo sending fails
        await update.message.reply_text(welcome_text, parse_mode='Markdown', reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the main command list."""
    help_text = (
        "**📚 B O T   C O M M A N D S**\n\n"
        "**Game:**\n"
        "• `/game` - Start a new pixel challenge (Normal Mode).\n"
        "• `/fastgame` - Start the 10-round fast challenge.\n"
        "• `/skip` - Skip the current normal game and reveal the answer.\n"
        "• `/hint` - Use a hint (costs points).\n\n"
        "**Economy & Stats:**\n"
        "• `/myscore` - Check your current point balance.\n"
        "• `/profile` or `/stats` - View your rank, streak, and album progress.\n"
        "• `/leaderboard` - See the global top players.\n"
        "• `/daily` - Claim your daily bonus points.\n\n"
        "**Collection & Shop:**\n"
        "• `/album [Letter]` - View your image categories. Use a letter (e.g., `/album A`) to filter your large collection.\n"
        "• `/shop` - See and buy special in-game items.\n\n"
        "Use `/howtoplay` for game details."
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='game_easy')],
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    
    await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=keyboard)

async def howtoplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed rules for the game."""
    rules_text = (
        "**📜 How To Play**\n\n"
        "**1. The Goal:** Guess the object, person, or word in the pixelated image.\n\n"
        "**2. Difficulty & Points (`/game`):**\n"
        "• **Easy (25x downscale):** Clearest image, low reward.\n"
        "• **Extreme (20x downscale):** Heavily pixelated, high reward.\n"
        "The harder the difficulty, the more points you earn!\n\n"
        "**3. Guessing:** Just send the word you think is correct in the chat. Case and spaces don't matter, e.g., if the answer is `Pineapple`, both `pineapple` and `PINE APPLE` will work.\n\n"
        "**4. Hints & Costs:**\n"
        "• You get one **free hint** (the category) with every game.\n"
        "• Use `/hint` to reveal a letter in the word (costs 10 points).\n\n"
        "**5. Fast Mode (`/fastgame`):**\n"
        "• 10 consecutive rounds.\n"
        "• **10 seconds** time limit per image.\n"
        "• No hints or skips. Guess correctly to immediately advance to the next round.\n\n"
        "Good luck, Agent!"
    )
    
    await update.message.reply_text(rules_text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows bot version and developer information."""
    process = psutil.Process(os.getpid())
    uptime = datetime.now() - start_time
    
    about_text = (
        f"**🤖 About {BOT_USERNAME}**\n\n"
        f"**Version:** v3.1 (Expanded Categories & Album)\n"
        f"**Developer:** Ankit / Pixel Team\n"
        f"**Source APIs:** Pexels, Unsplash, Google Gemini AI\n"
        f"**Uptime:** {str(timedelta(seconds=int(uptime.total_seconds())))}\n"
        f"**Messages Processed:** {total_messages_processed:,}\n\n"
        f"**Support Group:** [Join here]({SUPPORT_GROUP_LINK})"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Support Group", url=SUPPORT_GROUP_LINK)]
    ])
    
    await update.message.reply_text(about_text, parse_mode='Markdown', reply_markup=keyboard)

async def photoid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """For developer use: Gets the file_id of the last sent photo."""
    # This is often used by sending a photo and then sending /photoid as caption or reply.
    if update.effective_message.photo:
        file_id = update.effective_message.photo[-1].file_id # Get largest resolution
        await update.message.reply_text(f"File ID of the photo: `{file_id}`", parse_mode='Markdown')
    elif update.effective_message.reply_to_message and update.effective_message.reply_to_message.photo:
        file_id = update.effective_message.reply_to_message.photo[-1].file_id
        await update.message.reply_text(f"File ID of the replied photo: `{file_id}`", parse_mode='Markdown')
    else:
        await update.message.reply_text("Please send this command as a reply to a photo or as the caption of a photo to get its ID.")

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allows a user to skip the current normal game and reveals the answer (Updated)."""
    chat_id = update.effective_chat.id
    game_state = active_games.pop(chat_id, None)
    
    if chat_id in fast_games:
        await update.message.reply_text("You cannot skip Fast Mode. Use `/endfastgame` to quit.")
        return

    if game_state:
        correct_answer = game_state['answer'].upper()
        letter_count = len(game_state['answer']) 
        original_url = game_state['url']
        
        # UX Improvement: Add /game button (as requested)
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start New Game", callback_data='game_easy')]])
        
        await update.message.reply_text(
            f"🛑 **G A M E   S K I P P E D** 🛑\n The correct solution was: **{correct_answer}** ({letter_count} letters).",
            parse_mode='Markdown',
            reply_markup=keyboard # Add the inline keyboard
        )
        
        try:
            # Send the original image
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=f"Original Image. Solution: **{correct_answer}**.",
                parse_mode='Markdown'
            )
        except Exception:
            await update.message.reply_text("Could not send the original image file.", parse_mode='Markdown')
    else:
        await update.message.reply_text("No active normal game to skip. Use `/game` to start one.")

async def simple_hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides a category hint for the normal game."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    game_state = active_games.get(chat_id)
    if not game_state:
        await update.message.reply_text("No active normal game to get a hint for. Use `/game` to start one.")
        return

    points_cost = GAME_LEVELS[game_state['difficulty']]['hint_cost']
    max_hints = GAME_LEVELS[game_state['difficulty']]['max_hints']
    
    current_score = get_user_score(user_id)
    
    if game_state['hints_taken'] >= max_hints:
        await update.message.reply_text("You have already used the maximum number of hints for this game.")
        return

    if current_score < points_cost:
        await update.message.reply_text(
            f"❌ **INSUFFICIENT FUNDS!**\n"
            f"You need **{points_cost}** points for a hint, but you only have **{current_score}**.\n"
            f"Earn points by solving images or claim your `/daily` bonus."
            , parse_mode='Markdown'
        )
        return

    # Deduct points
    update_user_score(user_id, -points_cost)
    game_state['hints_taken'] += 1
    
    # Reveal a random unrevealed letter
    answer = game_state['answer'].lower()
    hint_list = list(game_state['hint_string'])
    
    unrevealed_indices = [i for i, char in enumerate(hint_list) if char == '_']
    
    if not unrevealed_indices:
        # Should not happen if the answer isn't fully revealed
        await update.message.reply_text("The word is already fully revealed!")
        return

    # Choose a random index to reveal
    index_to_reveal = random.choice(unrevealed_indices)
    hint_list[index_to_reveal] = answer[index_to_reveal].upper()
    game_state['hint_string'] = "".join(hint_list)
    
    # Send both the letter hint and the descriptive hint sentence
    new_score = get_user_score(user_id)
    
    hint_message = (
        f"💡 **HINT USED!** (-{points_cost} Points)\n"
        f"**New Progress**: `{game_state['hint_string']}`\n"
        f"**Clue**: *{game_state['hint_sentence']}*\n"
        f"**Hints Left**: **{max_hints - game_state['hints_taken']}**\n"
        f"**New Balance**: **{new_score}** Points."
    )
    
    await update.message.reply_text(hint_message, parse_mode='Markdown')


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompts user to select game difficulty (Updated)."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text("A normal game session is currently active. Please submit your guess or use `/skip`.")
        return
    
    if chat_id in fast_games:
        await update.message.reply_text("A fast game session is currently active. Please finish it or use `/skip`.")
        return

    is_extreme_available = PEXELS_API_KEY or UNSPLASH_ACCESS_KEY

    # Easy mode is now the clearest/easiest
    row1 = [
        InlineKeyboardButton(f"Easy (+{GAME_LEVELS['easy']['points']} pts)", callback_data='game_easy'),
        InlineKeyboardButton(f"Medium (+{GAME_LEVELS['medium']['points']} pts)", callback_data='game_medium'),
    ]
    
    row2 = [
        InlineKeyboardButton(f"Hard (+{GAME_LEVELS['hard']['points']} pts)", callback_data='game_hard'),
    ]
    
    if is_extreme_available:
        row2.insert(1, InlineKeyboardButton(f"Extreme (+{GAME_LEVELS['extreme']['points']} pts)", callback_data='game_extreme'))

    keyboard = [row1, row2]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "**📸 I N I T I A T E   G A M E**\n\nSelect a precision level for the image analysis (Difficulty):", 
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles difficulty selection, hint request, and end game (Updated)."""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    user_id = query.from_user.id
    data = query.data
    
    if data == 'game_end':
        # Logic for game end/skip (normal mode)
        game_state = active_games.pop(chat_id, None)
        
        if game_state:
            correct_answer = game_state['answer'].upper()
            letter_count = len(game_state['answer'])
            original_url = game_state['url']
            
            # UX Improvement: Add /game button (as requested)
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start New Game", callback_data='game_easy')]])
            
            try:
                 await query.edit_message_caption(
                    caption=f"🛑 **G A M E   T E R M I N A T E D** 🛑\n The correct solution was: **{correct_answer}** ({letter_count} letters).",
                    parse_mode='Markdown',
                    reply_markup=keyboard 
                )
            except Exception:
                await context.bot.send_message(chat_id, f"The game has ended. The correct answer was: **{correct_answer}** ({letter_count} letters).", parse_mode='Markdown', reply_markup=keyboard)
                
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=f"Original Image. Solution: **{correct_answer}**.",
                    parse_mode='Markdown'
                )
            except Exception:
                pass
        else:
            await context.bot.send_message(chat_id, "No active game to terminate.")
        return

    if data == 'game_hint':
        # Delegate to the command handler logic
        await simple_hint_command(query, context)
        return

    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, "A game is already active!")
        return
    
    # --- SPEED OPTIMIZATION: Send Loading Message First ---
    loading_message = await context.bot.send_message(
        chat_id, 
        f"**Challenge Initiated:** *{difficulty.upper()}*.\n\n"
        f"**⏳ Downloading and analyzing image. Please wait...**",
        parse_mode='Markdown'
    )
    
    # Fetch and pixelate image (This runs the heavy task)
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    if not pixelated_img_io:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text=f"❌ **Error**: Image acquisition failed. Details: {answer}",
                parse_mode='Markdown'
            )
        except Exception:
            await context.bot.send_message(chat_id, f"❌ **Error**: Image acquisition failed. Details: {answer}")
        return
        
    initial_hint_string = '_' * len(answer)
    
    active_games[chat_id] = {
        'answer': answer,               # AI Answer
        'difficulty': difficulty,
        'url': original_url,
        'hints_taken': 0,
        'hint_string': initial_hint_string,
        'category': category,           # Original search category (for album unlock)
        'hint_sentence': hint_sentence  # AI-generated descriptive hint
    }
    
    level_data = GAME_LEVELS[difficulty]
    points = level_data['points']
    
    caption = (
        f"**📸 V I S U A L   C H A L L E N G E: {difficulty.upper()}**\n\n" 
        f"Identify the object in this high-pixel density image.\n"
        f"**Reward**: **+{points} Points**\n"
        f"**Progress**: `{initial_hint_string}` (**{len(answer)}** letters)\n" 
        f"Use `/hint` for a complimentary category clue or click the button below."
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Request Letter Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - active_games[chat_id]['hints_taken']} remaining)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
    ])
    
    try:
        # Delete the loading message and send the photo
        await context.bot.delete_message(chat_id, loading_message.message_id)
        
        await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='Markdown',
            reply_markup=game_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send pixelated photo: {e}")
        await context.bot.send_message(chat_id, "❌ **Error**: Image transmission failed. Challenge cancelled.")
        del active_games[chat_id]


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Checks the user's guess against the active game's answer (Normal or Fast) (Updated)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    user_guess = text.lower().strip()
    
    # --- 1. Check Fast Game Mode ---
    if chat_id in fast_games:
        state = fast_games[chat_id]
        correct_answer = state['answer'].lower()
        
        # Check if the guess is an exact match or contains the correct answer as a word
        if user_guess == correct_answer or correct_answer in user_guess.split() or user_guess in correct_answer.split():
            
            # 1. Update Score and State
            reward = FAST_MODE_SETTINGS['reward']
            state['total_score'] += reward
            # Do not increment current_round here; it's handled by start_fast_round after the success message
            
            # 2. Cancel Timer (Crucial for fast mode speed)
            if 'timer_task' in state and state['timer_task'] and not state['timer_task'].done():
                state['timer_task'].cancel()

            # 3. Update the last message's caption to indicate success
            try:
                if 'last_message_id' in state:
                    await context.bot.edit_message_caption(
                        chat_id=chat_id,
                        message_id=state['last_message_id'],
                        caption=f"✅ **GUESSED!** The answer was **{correct_answer.upper()}**.",
                        reply_markup=None,
                        parse_mode='Markdown'
                    )
            except Exception:
                pass
                
            # 4. Send Success Message
            await context.bot.send_message(
                chat_id, 
                f"✅ **BINGO!** **+{reward}** Points. Correct Answer: **{correct_answer.upper()}**."
            )

            # 5. Advance to next round
            state['current_round'] += 1
            if state['current_round'] > state['max_rounds']:
                await end_fast_game(chat_id, context)
            else:
                await start_fast_round(chat_id, context)
                
        return
    
    # --- 2. Check Normal Game Mode ---
    game_state = active_games.get(chat_id)
    if not game_state:
        return 

    correct_answer = game_state['answer'].lower()
    
    if user_guess == correct_answer or correct_answer in user_guess.split() or user_guess in correct_answer.split():
        difficulty = game_state['difficulty']
        points = GAME_LEVELS[difficulty]['points']
        original_url = game_state['url']
        category = game_state['category']
        
        update_user_score(user_id, points)
        await save_solved_image(user_id, category) 
        
        del active_games[chat_id]
        
        current_score = get_user_score(user_id)
        letter_count = len(correct_answer) 
        
        caption = (
            f"✅ **S O L U T I O N   A C Q U I R E D !** ✅\n"
            f"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** ({letter_count} letters)\n\n"
            f"**Reward**: **+{points} Points**\n"
            f"**Current Balance**: **{current_score:,}**\n"
            f"View the original image below."
        )
        
        # UX Improvement: Add /game button (as requested)
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start New Game", callback_data='game_easy')]])
        
        try:
            # Send the original photo
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='Markdown',
                reply_markup=keyboard 
            )
        except Exception:
            await context.bot.send_message(chat_id, f"{caption}\n(Original image file unavailable).", parse_mode='Markdown', reply_markup=keyboard)
            
# ------------------------------------------------------------------
# --- SUPPORT COMMAND HANDLERS (Standard Implementations) ---
# ------------------------------------------------------------------

async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the user's current score."""
    user_id = update.effective_user.id
    score = get_user_score(user_id)
    await update.message.reply_text(f"💰 Your current score is: **{score:,} Points**.", parse_mode='Markdown')

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays user's full profile stats."""
    user = update.effective_user
    rank, score, solved_count, current_streak, last_daily_claim = await get_user_profile_data(user.id)

    profile_text = (
        f"**👤 Agent Profile: {user.first_name}**\n\n"
        f"**Rank**: #{rank:,}\n"
        f"**Score**: **{score:,}** Points\n"
        f"**Solved Images**: {solved_count:,}\n"
        f"**Current Streak**: {current_streak}\n\n"
        f"Use `/album` to see your collected categories."
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("View Album", callback_data='album_view_1')]
    ])
    
    await update.message.reply_text(profile_text, parse_mode='Markdown', reply_markup=keyboard)

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the global leaderboard."""
    top_scores = get_top_scores(limit=10)
    
    leaderboard_text = "**🏆 Global Leaderboard (Top 10)**\n\n"
    
    if not top_scores:
        leaderboard_text += "No scores recorded yet. Start a game with `/game`!"
    else:
        for i, (user_id, score) in enumerate(top_scores, 1):
            try:
                # Try to get user's name from context. If not available, use the ID (or 'Agent')
                member = await context.bot.get_chat_member(update.effective_chat.id, user_id)
                name = member.user.first_name
            except Exception:
                name = f"Agent {str(user_id)[:4]}..." # Fallback name
            
            leaderboard_text += f"{i}. **{name}** - **{score:,}** Points\n"

    await update.message.reply_text(leaderboard_text, parse_mode='Markdown')

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allows user to claim a daily bonus."""
    user_id = update.effective_user.id
    
    _, score, _, _, last_daily_claim = await get_user_profile_data(user_id)
    
    now_utc = datetime.now(pytz.utc)
    
    if last_daily_claim and (now_utc - last_daily_claim).total_seconds() < 24 * 3600:
        next_claim_time = last_daily_claim + timedelta(hours=24)
        time_left = next_claim_time - now_utc
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        
        await update.message.reply_text(
            f"⏳ You have already claimed your daily bonus.\n"
            f"Come back in **{hours}h {minutes}m** to claim your next **+{DAILY_BONUS_POINTS}** points!",
            parse_mode='Markdown'
        )
    else:
        success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
        if success:
            new_score = get_user_score(user_id)
            await update.message.reply_text(
                f"🎉 **DAILY BONUS CLAIMED!**\n"
                f"You received **+{DAILY_BONUS_POINTS}** points!\n"
                f"**New Balance**: **{new_score:,}** Points.",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("❌ Failed to claim daily bonus due to a database error.")

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays available items in the shop."""
    shop_text = "**🛒 Pixel Peep Shop**\n\n"
    keyboard_rows = []
    
    for key, item in SHOP_ITEMS.items():
        shop_text += f"**{key}. {item['name']}** - **{item['cost']}** Points\n"
        shop_text += f"*{item['description']}*\n\n"
        keyboard_rows.append([InlineKeyboardButton(f"Buy {item['name']} ({item['cost']})", callback_data=f'buy_{key}')])

    reply_markup = InlineKeyboardMarkup(keyboard_rows)
    current_score = get_user_score(update.effective_user.id)
    shop_text += f"**Your Balance**: **{current_score:,}** Points"
    
    await update.message.reply_text(shop_text, parse_mode='Markdown', reply_markup=reply_markup)

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles purchase of shop items via command or callback."""
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        data = query.data
        message = query.message
        user_id = query.from_user.id
        item_id = data.split('_')[1]
    else:
        message = update.message
        user_id = update.effective_user.id
        if not context.args or context.args[0] not in SHOP_ITEMS:
            await message.reply_text("Please specify a valid item ID. Use `/shop` to see items.")
            return
        item_id = context.args[0]
        
    item = SHOP_ITEMS.get(item_id)
    if not item:
        await message.reply_text("Invalid item ID.")
        return
        
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    if current_score < cost:
        await message.reply_text(
            f"❌ **Failed to Buy {item['name']}**\n"
            f"You need **{cost}** points but only have **{current_score}**."
            , parse_mode='Markdown'
        )
        return
        
    # --- Transaction Logic ---
    update_user_score(user_id, -cost)
    new_score = get_user_score(user_id)
    
    # NOTE: Actual item effect implementation is left for future development
    
    await message.reply_text(
        f"✅ **PURCHASE SUCCESSFUL!**\n"
        f"You bought **{item['name']}** for **{cost}** points.\n"
        f"*Functionality is currently limited.*\n"
        f"**New Balance**: **{new_score:,}** Points."
        , parse_mode='Markdown'
    )
    
async def album_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the user's solved image collection/album with segmentation."""
    user_id = update.effective_user.id
    # Fetch all categories, sorted alphabetically by the DB query (ORDER BY category ASC)
    all_collected_categories = await get_user_collection(user_id) 
    
    total_collected = len(all_collected_categories)
    total_categories = len(SEARCH_CATEGORIES)
    
    # Check for filtering argument (e.g., /album A)
    filter_letter = None
    if context.args and len(context.args[0]) == 1 and context.args[0].isalpha():
        filter_letter = context.args[0].upper()
        
    
    # 1. Filter the list
    if filter_letter:
        collected_categories = [cat for cat in all_collected_categories if cat.upper().startswith(filter_letter)]
        album_title = f"🖼️ Your Pixel Album (Categories starting with **{filter_letter}**)"
    elif total_collected > 20:
        # Default view for large album is the first 20 items
        collected_categories = all_collected_categories[:20]
        album_title = "🖼️ Your Pixel Album (First 20 Items)"
    else:
        # Default view for small album is all items
        collected_categories = all_collected_categories
        album_title = "🖼️ Your Pixel Album (All Collected Items)"


    album_text = f"**{album_title}**\n\n"
    
    if not collected_categories:
        if filter_letter:
            album_text += f"No categories starting with **{filter_letter}** found in your album."
        else:
            album_text += "Your album is empty! Solve images to collect categories.\nStart with `/game`."
    else:
        collected_map = {cat: SEARCH_CATEGORIES.get(cat, '❓') for cat in collected_categories}
        
        album_text += "**Collected Categories:**\n"
        
        # Display as a grid of emojis
        categories_display = ""
        count = 0
        for cat, emoji in collected_map.items():
            # Display format: Emoji + Category Name
            categories_display += f"{emoji} {cat.capitalize()} | "
            count += 1
            if count % 3 == 0: # 3 items per line for better readability with full names
                categories_display += "\n"
                
        album_text += categories_display.strip()
        
    
    album_text += f"\n\n**Total Progress**: **{total_collected}** / **{total_categories}** categories collected."

    # 2. Add segmentation instructions if the album is large AND not already filtered
    if total_collected > 20 and not filter_letter:
         album_text += (
             f"\n\n*Your collection is large! To view all items, use the segmented view:*\n"
             f"**Example**: `/album A` (for categories starting with A)\n"
             f"**Example**: `/album T` (for categories starting with T)"
         )
         
    
    if update.callback_query:
        await update.callback_query.message.reply_text(album_text, parse_mode='Markdown')
    else:
        await update.message.reply_text(album_text, parse_mode='Markdown')

async def donate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides donation information."""
    donate_text = (
        "**💖 Support {BOT_USERNAME} Development 💖**\n\n"
        "If you enjoy the bot, consider making a small donation to help cover server costs and fund future updates (like new game modes or better AI integration).\n\n"
        #"**UPI ID:** `{UPI_ID_FOR_DONATION}`\n\n"
        "You can scan the QR code below for easy payment. Thank you for your support!"
    )
    
    try:
        await update.message.reply_photo(
            photo=DONATION_QR_CODE_ID,
            caption=donate_text,
            parse_mode='Markdown'
        )
    except Exception:
        await update.message.reply_text(donate_text, parse_mode='Markdown')

async def handle_core_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general callbacks like help menu links."""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'help_menu':
        await howtoplay_command(query, context)
        # Try to edit the original message to remove the buttons
        try:
             await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
             pass
    
    if data.startswith('album_view_'):
        # For multi-page album view (now replaced by segmentation logic)
        await album_command(query, context)
        # Try to edit the original message to remove the buttons
        try:
             await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
             pass

# ------------------------------------------------------------------
# --- ADMIN & UTILITY FUNCTIONS ---
# ------------------------------------------------------------------

def check_owner_wrapper(handler):
    """Decorator to restrict command usage to the bot owner."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id == BROADCAST_ADMIN_ID:
            return await handler(update, context)
        else:
            await update.message.reply_text("❌ You are not authorized to use this command.")
    return wrapper
    
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message to all known chats (Owner-only)."""
    if not context.args:
        await update.message.reply_text("Usage: `/broadcast Your message here`")
        return

    message_to_send = " ".join(context.args)
    sent_count = 0
    fail_count = 0
    
    await update.message.reply_text(f"Starting broadcast to {len(known_users)} chats...")
    
    for chat_id in list(known_users): # Iterate over a copy
        try:
            await context.bot.send_message(chat_id, message_to_send, parse_mode='Markdown')
            sent_count += 1
        except Exception as e:
            logger.warning(f"Failed to send message to chat {chat_id}: {e}")
            fail_count += 1
            await asyncio.sleep(0.1) # Small delay to respect rate limits

    await update.message.reply_text(
        f"**✅ Broadcast Finished.**\n"
        f"**Sent**: {sent_count}\n"
        f"**Failed**: {fail_count}",
        parse_mode='Markdown'
    )


async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles a new user joining the chat."""
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            # Bot was added to a new chat
            save_chat_id(update.effective_chat.id)
            await context.bot.send_message(
                update.effective_chat.id,
                f"Thank you for adding me! Use `/start` to see a welcome message or `/game` to start the challenge. I need `Send Messages` permission to work correctly.",
            )
            return

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text messages (for game guessing)."""
    global total_messages_processed
    total_messages_processed += 1
    
    text = update.effective_message.text or update.effective_message.caption
    
    if not text or text.startswith('/'): # Ignore commands that weren't caught by handlers
        return
        
    await check_guess_and_update_score(update, context, text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice handler (Placeholder)."""
    if update.effective_chat.id in active_games or update.effective_chat.id in fast_games:
        await update.message.reply_text("Sorry, I currently cannot process voice messages for game answers. Please type your guess.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error."""
    logger.error("A critical error occurred:", exc_info=context.error)
    # Optional: Send a user-friendly error message if an update object exists
    if update and update.effective_chat:
        try:
             await context.bot.send_message(
                update.effective_chat.id,
                "⚠️ **An unexpected error occurred.** The developer has been notified. Please try again or use `/help`."
                , parse_mode='Markdown'
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# --- MAIN EXECUTION ---
# ------------------------------------------------------------------

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN is missing!")
        return
        
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # 1. Connect to DB and load data
    conn, error = db_connect()
    if not conn:
        logger.critical(f"CRITICAL: Failed to connect to PostgreSQL on startup. {error}. Check your DATABASE_URL.")
    
    load_known_users()
    
    if not PEXELS_API_KEY and not UNSPLASH_ACCESS_KEY:
        logger.warning("WARNING: Both PEXELS_API_KEY and UNSPLASH_ACCESS_KEY are missing. Image guessing game will not work.")
    
    if not GEMINI_API_KEY:
        logger.warning("WARNING: GEMINI_API_KEY is missing. AI-based answer generation will fail.")

    # 2. Add Handlers
    
    # Core Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("stats", profile_command)) 
    application.add_handler(CommandHandler("profile", profile_command))

    # Shop/Economy Commands
    application.add_handler(CommandHandler("shop", shop_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("donate", donate_command)) 

    # Game Commands
    application.add_handler(CommandHandler("game", game_command))
    application.add_handler(CommandHandler("fastgame", fastgame_command)) # NEW FAST MODE
    application.add_handler(CommandHandler("myscore", my_score_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("album", album_command)) 
    application.add_handler(CommandHandler("collection", album_command)) 
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("hint", simple_hint_command))
    application.add_handler(CommandHandler("howtoplay", howtoplay_command)) 
    application.add_handler(CommandHandler("photoid", photoid_command)) 
    
    # Owner-Only Command
    application.add_handler(CommandHandler("broadcast", check_owner_wrapper(broadcast_command))) 
    application.add_handler(CommandHandler("endfastgame", end_fast_game)) # Utility to manually stop a fast game

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
    # The process_message handler will now also catch the /photoid command usage inside its logic
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION), 
        process_message
    ))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Callback Query Handler 
    application.add_handler(CallbackQueryHandler(handle_game_callback, pattern=r'^(game_|game_end)'))
    application.add_handler(CallbackQueryHandler(handle_core_callback, pattern=r'^(help_menu|album_view_)'))
    application.add_handler(CallbackQueryHandler(buy_command, pattern=r'^buy_')) 

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
