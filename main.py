# main.py (Final, Standardised, Premium Look, with Gemini AI Integration)

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
# --- NEW: GEMINI INTEGRATION ---
try:
    import google.genai
    from google.genai import types
except ImportError:
    # Fallback in case of missing library (though user should install it)
    print("WARNING: 'google-genai' not installed. AI features will be disabled.")

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # NEW: Fetch Gemini Key

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
# Structure: {chat_id: {'answer': 'keyword', 'difficulty': 'easy', 'url': 'original_url', 'hints_taken': 0, 'hint_string': '_______', 'category': 'animal', 'hint_sentence': '...'}}
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
# --- CONSTANTS & CONFIGURATIONS (Standardised for Premium Look) ---
# ------------------------------------------------------------------

BOT_USERNAME = "Pixel Peep" 
SUPPORT_GROUP_LINK = "https://t.me/pixel_studis" 
MAIN_GROUP_LINK = "https://t.me/+JNpAXIuwHTIyOWY1" 
WELCOME_IMAGE_URL = "AgACAgUAAxkBAAPNaSdU6DYWrJbMvkAsoRF93H3V2x8AAhYLaxsfjEFVReKLNrpOrBUBAAMCAAN4AAM2BA" 

# ADJUSTED PIXELATION LEVELS: Pixeling increased for Hard, Medium, and Easy.
GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 3, 'hint_cost': 10}, 
    'hard': {'downscale_factor': 15, 'points': 70, 'max_hints': 3, 'hint_cost': 10}, 
    'medium': {'downscale_factor': 10, 'points': 50, 'max_hints': 3, 'hint_cost': 10}, 
    'easy': {'downscale_factor': 5, 'points': 30, 'max_hints': 3, 'hint_cost': 10} 
}

# Categories for Album/Stickers (Used only for image search, not for answer generation)
SEARCH_CATEGORIES = {
    'nature': '🌳', 'city': '🏙️', 'animal': '🐾', 'food': '🍕', 
    'travel': '✈️', 'object': '💡', 'landscape': '🏞️', 'mountain': '⛰️', 
    'beach': '🏖️', 'technology': '🤖'
}
search_queries = list(SEARCH_CATEGORIES.keys())

# Shop Items (Flashlight now reveals 2 letters)
SHOP_ITEMS = {
    '1': {'name': 'Flashlight', 'cost': 200, 'description': 'Reveal 2 random unrevealed letters in the word.'},
    '2': {'name': 'First Letter', 'cost': 100, 'description': 'Reveal the first letter of the answer.'}
}

DAILY_BONUS_POINTS = 50

# --- REMOVED: HINT_SENTENCES (AI will generate the hint) ---
# ------------------------------------------------------------------
# --- POSTGRESQL/NEON DB CONNECTION AND DATA MANAGEMENT FUNCTIONS ---
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
            cur.execute("SELECT category FROM user_collection WHERE user_id = %s;", (user_id,))
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
# --- NEW: GEMINI AI INTEGRATION FUNCTIONS ---
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
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
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
    # raw_answer is no longer needed, use query as a placeholder
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
    # raw_answer is no longer needed, use query as a placeholder
    return image_url, query, "Unsplash"

# Main Image Processing Function (MODIFIED: TO USE AI ANSWER/HINT)
async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None, str | None, str | None]:
    """Randomly selects an API, fetches, sends to AI, and pixelates. Returns (io, AI_answer, url, category, AI_hint_sentence)."""
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
        
        # 2. NEW: Get AI Answer and Hint
        ai_answer, hint_sentence = await get_ai_answer_and_hint(image_data)
        
        if not ai_answer:
            return None, "AI failed to generate a valid, one-word answer.", None, None, None
        
        # 3. Process/Pixelate Image using PIL
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        downscale_factor = GAME_LEVELS[difficulty]['downscale_factor']
        
        width, height = img.size
        # ... (rest of pixelation logic remains the same)
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
        
        # Return the AI-generated answer and hint
        return output, ai_answer, image_url, category, hint_sentence
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image Download Error: {e}")
        return None, f"Image Download Error: {e}", None, None, None
    except Exception as e:
        logger.error(f"Image Processing/Unknown Error: {e}")
        return None, f"Image Processing/Unknown Error: {e}", None, None, None


# ------------------------------------------------------------------
# --- PLAYER PROFILE AND SHOP COMMANDS (Standardised) ---
# ------------------------------------------------------------------
# ... (profile_command, album_command, shop_command, buy_command, daily_command remain the same) ...
async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user their personal report card in a standard format."""
    user = update.effective_user
    rank, points, solved_count, current_streak, last_daily_claim = await get_user_profile_data(user.id)
    
    if rank is None and isinstance(last_daily_claim, str):
        await update.message.reply_text("❌ Error accessing player profile data. Please try again later.")
        return

    profile_text = (
        "👤 **P L A Y E R   P R O F I L E**\n\n"
        f"**Player ID**: `{user.id}`\n"
        f"**Username**: **{user.first_name}**\n\n"
        "**— S T A T I S T I C S —**\n"
        f"**Global Rank**: **#{rank}**\n"
        f"**Current Balance**: `{points:,}` Points\n"
        f"**Solved Images**: `{solved_count:,}`\n"
        f"**Active Streak**: `{current_streak}` Days\n\n"
        "View the full rankings using `/leaderboard`."
    )
    await update.message.reply_text(profile_text, parse_mode='Markdown')

async def album_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows only the collected stickers/images in a standard format."""
    user_id = update.effective_user.id
    collected_categories = await get_user_collection(user_id)
    
    album_text = "🖼️ **C O L L E C T I O N   A L B U M** 🖼️\n\n"
    unlocked_count = len(collected_categories)
    total_count = len(SEARCH_CATEGORIES)
    
    album_text += f"**Collection Status**: **{unlocked_count}/{total_count}** Categories Unlocked.\n\n"

    if not collected_categories:
        album_text += "❌ **ALBUM IS EMPTY!**\n"
        album_text += "Start a challenge with `/game` to unlock your first category!"
    else:
        album_text += "**— U N L O C K E D   C A T E G O R I E S —**\n"
        
        sorted_collected = sorted(collected_categories)
        display_list = []
        for cat_name in sorted_collected:
            emoji = SEARCH_CATEGORIES.get(cat_name, '❓')
            display_list.append(f"• {emoji} **{cat_name.upper()}**")
        
        album_text += "\n".join(display_list)
    
    await update.message.reply_text(album_text, parse_mode='Markdown')

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Opens the shop menu (Standardised)."""
    user_id = update.effective_user.id
    current_score = get_user_score(user_id)
    
    shop_text = f"🛒 **P I X E L   U T I L I T Y   S H O P**\n\n"
    shop_text += f"**Your Balance**: `{current_score:,}` Points\n\n"
    keyboard = []
    
    for item_id, item in SHOP_ITEMS.items():
        shop_text += f"**Item {item_id}: {item['name'].upper()}**\n"
        shop_text += f"  - *Cost*: **{item['cost']} Points**\n"
        shop_text += f"  - *Description*: {item['description']}\n\n"
        keyboard.append([InlineKeyboardButton(f"PURCHASE: {item['name']} ({item['cost']} pts)", callback_data=f'buy_{item_id}')])
        
    shop_text += "Use the button below or type `/buy <id>` to proceed with purchase."
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(shop_text, parse_mode='Markdown', reply_markup=reply_markup)

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles shop purchases."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        try:
            item_id = query.data.split('_')[1]
        except IndexError:
            await context.bot.send_message(user_id, "Invalid buy request.")
            return
    elif context.args and context.args[0] in SHOP_ITEMS:
        item_id = context.args[0]
    else:
        await update.message.reply_text("Invalid item ID. Use: `/buy 1` or `/buy 2`.")
        return

    item = SHOP_ITEMS[item_id]
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    if current_score < cost:
        response = f"❌ **TRANSACTION DENIED.** Insufficient balance. Required: **{cost} pts**, Available: **{current_score} pts**."
        if update.callback_query:
            await context.bot.send_message(update.effective_chat.id, response, parse_mode='Markdown')
        else:
            await update.message.reply_text(response, parse_mode='Markdown')
        return

    # Process purchase
    update_user_score(user_id, -cost)
    new_score = get_user_score(user_id)
    
    response = f"✅ **PURCHASE SUCCESSFUL!**\n*{item['name'].upper()}* activated.\n\n"
    
    if item_id == '1': # Flashlight (Reveal 2 letters)
        game_state = active_games.get(chat_id)
        if game_state and '_' in game_state['hint_string']:
            letters_to_reveal = 2 
            answer_word = game_state['answer']
            hint_list = list(game_state['hint_string'])
            
            for _ in range(letters_to_reveal):
                masked_indices = [i for i, char in enumerate(hint_list) if char == '_']
                if not masked_indices:
                    break
                    
                reveal_index = random.choice(masked_indices)
                hint_list[reveal_index] = answer_word[reveal_index].upper()
            
            game_state['hint_string'] = "".join(hint_list)
            
            response += f"**EFFECT**: Revealed {letters_to_reveal} characters.\n**Current State**: `{game_state['hint_string']}`"
            
            if '_' not in game_state['hint_string']:
                del active_games[chat_id]
                response += "\n\n**STATUS**: Word fully revealed. Game terminated (No points awarded)."
        else:
            response += "⚠️ **STATUS**: No active game or word is already complete."
    
    elif item_id == '2': # First Letter
        game_state = active_games.get(chat_id)
        if game_state and game_state['hint_string'][0] == '_':
            answer_word = game_state['answer']
            
            hint_list = list(game_state['hint_string'])
            hint_list[0] = answer_word[0].upper()
            game_state['hint_string'] = "".join(hint_list)
            
            response += f"**EFFECT**: The initial character is now visible.\n**Current State**: `{game_state['hint_string']}`"
        else:
            response += "⚠️ **STATUS**: No active game or initial character is already revealed."
            
    response += f"\n\n**New Balance**: **{new_score} Points**."
            
    if update.callback_query:
        await context.bot.send_message(update.effective_chat.id, response, parse_mode='Markdown')
    else:
        await update.message.reply_text(response, parse_mode='Markdown')


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allows a user to claim free points once every 24 hours (Standardised)."""
    user = update.effective_user
    user_id = user.id
    current_time = datetime.now(pytz.utc) 
    
    rank, score, solved_count, current_streak, last_claim = await get_user_profile_data(user_id)
    
    if rank is None and isinstance(last_claim, str):
        await update.message.reply_text("❌ Error accessing player profile data. Please try again later.")
        return
    
    if last_claim:
        if last_claim.tzinfo is None:
            last_claim = last_claim.replace(tzinfo=pytz.utc)
        
        next_claim_time = last_claim + timedelta(hours=24) 
        
        if current_time < next_claim_time:
            time_remaining = next_claim_time - current_time
            total_seconds = time_remaining.total_seconds()
            
            if total_seconds > 0:
                hours, remainder = divmod(int(total_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
            else:
                hours, minutes = 0, 0
            
            await update.message.reply_text(
                f"⏳ **DAILY CREDIT ACQUISITION**\n\n"
                f"You have already claimed today's credit. Next availability in: **{hours}h {minutes}m**.",
                parse_mode='Markdown'
            )
            return

    success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
    
    if success:
        new_score = get_user_score(user_id)
        await update.message.reply_text(
            f"💰 **DAILY CREDIT GRANTED!**\n\n"
            f"Acquired **+{DAILY_BONUS_POINTS}** Points.\n"
            f"**Current Balance**: **{new_score:,}** Points.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text("❌ Failed to process daily credit. Database error.")

# ------------------------------------------------------------------
# --- GAME COMMAND HANDLERS (Including New Commands) ---
# ------------------------------------------------------------------
# ... (howtoplay_command, photoid_command, skip_command remain the same, but with LETTER COUNT added to skip reveal) ...

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allows a user to skip the current game and reveals the answer (MODIFIED: Added Letter Count)."""
    chat_id = update.effective_chat.id
    game_state = active_games.pop(chat_id, None)
    
    if game_state:
        correct_answer = game_state['answer'].upper()
        letter_count = len(game_state['answer']) # NEW: Letter count
        original_url = game_state['url']
        
        await update.message.reply_text(
            f"🛑 **G A M E   S K I P P E D** 🛑\n The correct solution was: **{correct_answer}** ({letter_count} letters).",
            parse_mode='Markdown'
        )
        
        try:
            # Send the original image
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=f"Original Image. Solution: **{correct_answer}**.",
                parse_mode='Markdown'
            )
        except:
            await update.message.reply_text("Could not send the original image file.", parse_mode='Markdown')
    else:
        await update.message.reply_text("No active game to skip.")


async def simple_hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    MODIFIED: Provides the AI-generated descriptive hint and the letter count.
    Note: This is the complimentary hint, separate from the paid 'Request Assistance' (game_hint)
    """
    chat_id = update.effective_chat.id
    game_state = active_games.get(chat_id)
    
    if not game_state:
        await update.message.reply_text("No active challenge to provide a complimentary hint for.")
        return
        
    # NEW: Get AI-generated hint and answer length
    hint = game_state.get('hint_sentence', "The AI clue is unavailable for this round.")
    answer_length = len(game_state['answer'])
    
    hint_text = (
        f"💡 **C O M P L I M E N T A R Y   C L U E** 💡\n\n"
        f"**Word Length**: The answer has **{answer_length}** letters.\n"
        f"**Clue**: *{hint}*"
    )
    
    await update.message.reply_text(hint_text, parse_mode='Markdown')

async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompts user to select game difficulty."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text("A game session is currently active. Please submit your guess or use `/skip`.")
        return

    is_extreme_available = PEXELS_API_KEY or UNSPLASH_ACCESS_KEY

    row1 = [
        InlineKeyboardButton(f"Hard (+{GAME_LEVELS['hard']['points']} pts)", callback_data='game_hard'),
    ]
    
    if is_extreme_available:
        row1.insert(0, InlineKeyboardButton(f"Extreme (+{GAME_LEVELS['extreme']['points']} pts)", callback_data='game_extreme'))
    
    row2 = [
        InlineKeyboardButton(f"Medium (+{GAME_LEVELS['medium']['points']} pts)", callback_data='game_medium'),
        InlineKeyboardButton(f"Easy (+{GAME_LEVELS['easy']['points']} pts)", callback_data='game_easy'),
    ]

    keyboard = [row1, row2]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "**📸 I N I T I A T E   G A M E**\n\nSelect a precision level for the image analysis:", 
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles difficulty selection, hint request, and end game (MODIFIED: To store/use AI data)."""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    user_id = query.from_user.id
    data = query.data
    
    if data == 'game_end':
        game_state = active_games.pop(chat_id, None)
        if game_state:
            correct_answer = game_state['answer'].upper()
            letter_count = len(game_state['answer']) # NEW: Letter count
            original_url = game_state['url']
            
            try:
                 await query.edit_message_caption(
                    caption=f"🛑 **G A M E   T E R M I N A T E D** 🛑\n The correct solution was: **{correct_answer}** ({letter_count} letters).",
                    parse_mode='Markdown',
                    reply_markup=None
                )
            except:
                await context.bot.send_message(chat_id, f"The game has ended. The correct answer was: **{correct_answer}** ({letter_count} letters).", parse_mode='Markdown')
                
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=f"Original Image. Solution: **{correct_answer}**.",
                    parse_mode='Markdown'
                )
            except:
                pass
        else:
            await context.bot.send_message(chat_id, "No active game to terminate.")
        return

    if data == 'game_hint':
        game_state = active_games.get(chat_id)
        if not game_state:
            await context.bot.send_message(chat_id, "No active game to request assistance for.")
            return

        level_data = GAME_LEVELS[game_state['difficulty']]
        
        if game_state['hints_taken'] >= level_data['max_hints']:
            await context.bot.send_message(chat_id, "Maximum assistance quota reached for this challenge.")
            return

        hint_cost = level_data['hint_cost']
        current_score = get_user_score(user_id)
        
        if current_score < hint_cost:
             await context.bot.send_message(chat_id, f"❌ **DENIED.** Insufficient balance. Need **{hint_cost} pts**.")
             return
             
        update_user_score(user_id, -hint_cost) 
        game_state['hints_taken'] += 1
        
        answer_word = game_state['answer']
        current_hint_string = game_state['hint_string']
        
        masked_indices = [i for i, char in enumerate(current_hint_string) if char == '_']
        
        if masked_indices:
            reveal_index = random.choice(masked_indices)
            
            hint_list = list(current_hint_string)
            hint_list[reveal_index] = answer_word[reveal_index].upper()
            game_state['hint_string'] = "".join(hint_list)
        
        if '_' not in game_state['hint_string']:
            del active_games[chat_id]
            await context.bot.send_message(
                chat_id, 
                f"💡 **ASSISTANCE LOGGED.** -**{hint_cost} points** deducted.\n\nThe solution **{answer_word.upper()}** has been fully compromised by assistance. No points awarded.",
                parse_mode='Markdown'
            )
            return

        new_score = get_user_score(user_id)
        
        game_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"💡 Request Assistance (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - game_state['hints_taken']} remaining)", callback_data='game_hint')],
            [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
        ])

        await context.bot.send_message(
            chat_id, 
            f"💡 **ASSISTANCE PROVIDED.** -**{hint_cost} points** deducted. Balance: **{new_score}**\n\n**Progress**: `{game_state['hint_string']}`\n",
            parse_mode='Markdown',
            reply_markup=game_keyboard
        )
        return

    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, "A game is already active!")
        return
    
    try:
        await query.edit_message_text(f"**Challenge Initiated:** *{difficulty.upper()}*. Please wait for the image.", parse_mode='Markdown', reply_markup=None)
    except:
        await context.bot.send_message(chat_id, f"**Challenge Initiated:** *{difficulty.upper()}*. Please wait for the image.", parse_mode='Markdown')
    
    # MODIFIED: New return signature
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    if not pixelated_img_io:
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
    
    # NEW: Caption includes word length
    caption = (
        f"**📸 V I S U A L   C H A L L E N G E: {difficulty.UPPER()}**\n\n"
        f"Identify the object in this high-pixel density image.\n"
        f"**Reward**: **+{points} Points**\n"
        f"**Progress**: `{initial_hint_string}` (**{len(answer)}** letters)\n" # NEW: Letter Count in caption
        f"Use `/hint` for a complimentary category clue."
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Request Assistance (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - active_games[chat_id]['hints_taken']} remaining)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
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
        await context.bot.send_message(chat_id, "❌ **Error**: Image transmission failed. Challenge cancelled.")
        del active_games[chat_id]


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Checks the user's guess against the active game's answer (MODIFIED: Added Letter Count to win message)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    game_state = active_games.get(chat_id)
    if not game_state:
        return 

    correct_answer = game_state['answer'].lower()
    user_guess = text.lower().strip()
    
    # Check if the guess is the full answer, or if the answer is a subset of the guess, or vice-versa
    if user_guess == correct_answer or correct_answer in user_guess.split() or user_guess in correct_answer.split():
        difficulty = game_state['difficulty']
        points = GAME_LEVELS[difficulty]['points']
        original_url = game_state['url']
        category = game_state['category']
        
        update_user_score(user_id, points)
        await save_solved_image(user_id, category) 
        
        del active_games[chat_id]
        
        current_score = get_user_score(user_id)
        letter_count = len(correct_answer) # NEW: Letter count
        
        caption = (
            f"✅ **S O L U T I O N   A C Q U I R E D !** ✅\n"
            f"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** ({letter_count} letters)\n\n" # NEW: Letter Count
            f"**Reward**: **+{points} Points**\n"
            f"**Current Balance**: **{current_score:,}**\n"
            f"View the original image below."
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='Markdown'
            )
        except Exception:
            await context.bot.send_message(chat_id, f"{caption}\n(Original image file unavailable).", parse_mode='Markdown')
            
# ... (my_score_command, leaderboard_command, handle_core_callback, start_command, help_command, about_command, check_owner_wrapper, broadcast_command, new_chat_member_handler, process_message, handle_voice_message, error_handler remain the same) ...
async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user's current score (Standardised)."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    score = get_user_score(user_id)
    await update.message.reply_text(
        f"**{user_name}**'s current account balance: **{score:,}** Points. 🏅",
        parse_mode='Markdown'
    )

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the top 10 scores in a clean, professional format."""
    top_scores = get_top_scores(10)
    
    if not top_scores:
        await update.message.reply_text("The Global Leaderboard is currently being established. Play a game to secure your rank.")
        return
        
    leaderboard_text = "👑 **G L O B A L   L E A D E R B O A R D** 👑\n"
    leaderboard_text += "```\n"
    leaderboard_text += "RANK | PLAYER NAME            | SCORE\n"
    leaderboard_text += "---------------------------------------\n"
    
    for i, (user_id, score) in enumerate(top_scores, 1):
        try:
            user = await context.bot.get_chat(user_id)
            user_name = user.full_name[:20].ljust(20) 
        except Exception:
            user_name = f"User {user_id}"[:20].ljust(20)

        rank_str = str(i).rjust(4)
        score_str = f"{score:,}".rjust(5)
        
        leaderboard_text += f" {rank_str} | {user_name} | {score_str}\n"
        
    leaderboard_text += "```\n"
    leaderboard_text += "\n*Your ranking reflects your precision and dedication.*"
        
    await update.message.reply_text(leaderboard_text, parse_mode='Markdown')

async def handle_core_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general inline button presses like 'help_menu' and 'album_view' status."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == 'help_menu':
        await help_command(update, context) 
        
    elif data.startswith('album_view_'):
        category = data.split('_')[2]
        user_id = query.from_user.id
        collected_categories = await get_user_collection(user_id)
        
        if category in collected_categories:
            await query.answer(f"✅ Category UNLOCKED: {category.upper()}", show_alert=True)
        else:
            await query.answer(f"❌ Category PENDING: {category.upper()}", show_alert=True)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The premium, image-enhanced start command."""
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name
    save_chat_id(chat_id)
    
    start_message = (
        f"**W E L C O M E, {user_name.upper()}!**\n\n"
        "**Pixel Peep** is your premium image guessing utility. "
        "Engage in challenging visual quizzes, accumulate points, and dominate the global leaderboard.\n\n"
        "**— E N G A G E —**\n"
        "▸ Start a game using `/game` and select a difficulty.\n"
        "▸ Check your progress and rank with `/profile`.\n"
        "▸ Access powerful hints and tools via the `/shop`.\n\n"
        "Ready to test your visual precision?"
    )
    
    keyboard = [
        [InlineKeyboardButton("▶️ Initiate Game Challenge", callback_data='game_easy')],
        
        [InlineKeyboardButton("➕ Add to Your Group", url=f"https://t.me/{BOT_USERNAME}?startgroup=true")],
        [InlineKeyboardButton("💬 Support", url=SUPPORT_GROUP_LINK),
         InlineKeyboardButton("📢 Group", url=MAIN_GROUP_LINK)],
        
        [InlineKeyboardButton("📚 View All Commands", callback_data='help_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=WELCOME_IMAGE_URL,
            caption=start_message, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    except Exception as e:
        logger.error(f"Failed to send welcome photo. Sending text fallback: {e}")
        await update.message.reply_text(
            start_message, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the standard, professional help menu."""
    
    help_text = (
        "⚙️ **P I X E L  P E E P   C O M M A N D S** ⚙️\n\n"
        "**I. Core & Profile**\n"
        "`/start` - System initialization.\n"
        "`/profile` - Access player statistics and rank.\n"
        "`/leaderboard` - View top global players.\n"
        "`/album` - Review **unlocked** image categories.\n"
        "`/howtoplay` - Learn the rules and mechanics of the game.\n\n"
        
        "**II. Challenge & Game Control**\n"
        "`/game` - Begin a new visual challenge.\n"
        "`/skip` - Terminate the current challenge.\n"
        "`/hint` - Request a complimentary **descriptive** clue (includes word length).\n\n"
        
        "**IV. Utility & Economy**\n"
        "`/daily` - Acquire daily point credit.\n"
        "`/shop` - Access the utility shop for power-ups.\n"
        "`/buy <id>` - Execute a shop purchase.\n"
        "`/photoid` - Get the File ID of a photo/sticker.\n"
    )
        
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=help_text, 
        parse_mode='Markdown', 
        disable_web_page_preview=True
    )
    logger.info(f"[{update.effective_chat.id}] /help command used.")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    about_text = (
        "✨ **A B O U T   P I X E L  P E E P** ✨\n\n"
        "PixelBot is designed to deliver a premium, engaging image recognition experience. "
        "It features a robust game engine, persistent player statistics, and an integrated economy.\n\n"
        "**Version**: *1.0 (Gemini AI & Letter Count Update)*"
    )
    await update.message.reply_text(about_text, parse_mode='Markdown')

def check_owner_wrapper(handler):
    """Decorator to restrict access to the primary bot owner (BROADCAST_ADMIN_ID)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if user_id == str(BROADCAST_ADMIN_ID):
            await handler(update, context)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Access Denied. This utility is restricted to the bot administrator.")
    return wrapper
    
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Broadcasts the message the admin replies to."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    message_to_broadcast = update.message.reply_to_message
    text_content = None
    
    if message_to_broadcast:
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"<b>Broadcast Service Initiated.</b> Target count: {len(known_users)} chats...\nMode: Forwarding replied message.",
            parse_mode='HTML'
        )
    elif context.args:
        text_content = " ".join(context.args)
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"<b>Broadcast Service Initiated.</b> Target count: {len(known_users)} chats...\nMode: Sending plain text from arguments.",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("Syntax Error: Please **reply** to the message (photo, file, text, sticker, etc.) you want to broadcast, OR use `/broadcast <text message>`.", parse_mode='Markdown')
        return
        
    logger.info(f"[{user_id}] Initiating broadcast. Type: {'Forward' if message_to_broadcast else 'Text'}")

    sent_count = 0
    failed_count = 0
    
    chats_to_broadcast = list(known_users)
    
    for target_chat_id_str in chats_to_broadcast:
        target_chat_id = int(target_chat_id_str)
        try:
            if message_to_broadcast:
                await context.bot.forward_message(
                    chat_id=target_chat_id, 
                    from_chat_id=chat_id, 
                    message_id=message_to_broadcast.message_id
                )
            elif text_content:
                await context.bot.send_message(
                    chat_id=target_chat_id, 
                    text=text_content, 
                    parse_mode='Markdown' 
                )
            
            sent_count += 1
            await asyncio.sleep(0.1) 
        except Exception as e:
            failed_count += 1
            logger.warning(f"Broadcast failure for chat {target_chat_id}: {e}")

    final_message = f"<b>Broadcast Service Completed!</b>\nTransmission Success: <code>{sent_count}</code>\nTransmission Failed: <code>{failed_count}</code>"
    await context.bot.send_message(chat_id=chat_id, text=final_message, parse_mode='HTML')
    logger.info(f"[{user_id}] Broadcast completed. Sent: {sent_count}, Failed: {failed_count}")


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
        
    if text.startswith("/photoid"):
        await photoid_command(update, context)
        return 

    await check_guess_and_update_score(update, context, text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice handler (Now just a placeholder)."""
    pass

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error."""
    logger.error("A critical error occurred:", exc_info=context.error)

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

    # Game Commands
    application.add_handler(CommandHandler("game", game_command))
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

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
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
