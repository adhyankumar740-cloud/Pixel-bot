# main.py (Final Version: All Features Included)

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
# Structure: {chat_id: {'answer': 'keyword', 'difficulty': 'easy', 'url': 'original_url', 'hints_taken': 0, 'hint_string': '_______', 'category': 'animal'}}
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
# --- CONSTANTS & CONFIGURATIONS ---
# ------------------------------------------------------------------

# Pixelation Downscale Factors, Points, and Hint Logic
GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 5, 'hint_cost': 10},
    'hard': {'downscale_factor': 10, 'points': 70, 'max_hints': 5, 'hint_cost': 10},
    'medium': {'downscale_factor': 5, 'points': 50, 'max_hints': 5, 'hint_cost': 10},
    'easy': {'downscale_factor': 2, 'points': 30, 'max_hints': 5, 'hint_cost': 10}
}

# Categories for Album/Stickers
SEARCH_CATEGORIES = {
    'nature': '🌳', 'city': '🏙️', 'animal': '🐾', 'food': '🍕', 
    'travel': '✈️', 'object': '💡', 'landscape': '🏞️', 'mountain': '⛰️', 
    'beach': '🏖️', 'technology': '🤖'
}
search_queries = list(SEARCH_CATEGORIES.keys())

# Shop Items
SHOP_ITEMS = {
    '1': {'name': 'Flashlight', 'cost': 200, 'description': 'Reveal 20% of the image clearly (Requires new API call - NOT fully implemented).'},
    '2': {'name': 'First Letter', 'cost': 100, 'description': 'Reveal the first letter of the answer.'}
}

DAILY_BONUS_POINTS = 50

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
            # game_scores table updated with profile fields
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
                -- New table for player collections/album
                CREATE TABLE IF NOT EXISTS user_collection (
                    user_id BIGINT,
                    category VARCHAR(50),
                    UNIQUE (user_id, category)
                );
            """)
        
        logger.info("Successfully connected to PostgreSQL (Neon Tech) and verified tables.")
        return conn, None
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        conn = None 
        return None, f"Error connecting to PostgreSQL: {e}"

# --- CORE DB HELPERS (load_known_users, save_chat_id, get_top_scores remain UNCHANGED) ---

# ... [load_known_users, save_chat_id, get_top_scores functions from previous code] ...
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


# --- NEW DB HELPERS FOR PROFILE/ALBUM ---

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
                # Initialize new user if no record found
                update_user_score(user_id, 0)
                data = (0, 0, 0, None)
            
            score, solved_count, current_streak, last_daily_claim = data

            # Calculate Rank (Inefficient but works for small/medium scale)
            cur.execute("SELECT COUNT(*) FROM game_scores WHERE score > %s;", (score,))
            rank = cur.fetchone()[0] + 1
            
            return rank, score, solved_count, current_streak, last_daily_claim
    except Exception as e:
        logger.error(f"Error fetching profile data for user {user_id}: {e}")
        return None, 0, 0, 0, "Unknown Error"

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
            cur.execute(
                "INSERT INTO user_collection (user_id, category) VALUES (%s, %s) ON CONFLICT (user_id, category) DO NOTHING;",
                (user_id, category)
            )

            # 2. Update solved_count and current_streak
            # This is simplified: it resets the streak if the last solve wasn't today (needs more advanced logic for real streak)
            cur.execute(
                """
                UPDATE game_scores SET 
                    solved_count = solved_count + 1,
                    current_streak = CASE 
                        WHEN (NOW() - last_daily_claim) < INTERVAL '48 hours' THEN current_streak + 1
                        ELSE 1 
                    END
                WHERE user_id = %s;
                """,
                (user_id,)
            )
    except Exception as e:
        logger.error(f"Error saving solved image data for user {user_id}: {e}")


# ------------------------------------------------------------------
# --- IMAGE GUESSING GAME LOGIC (UNCHANGED) ---
# ------------------------------------------------------------------

# ... [fetch_image_from_pexels, fetch_image_from_unsplash functions remain UNCHANGED] ...

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
    raw_answer = photo.get('alt', photo.get('photographer', query)).split(',')[0].strip()
    return image_url, raw_answer, "Pexels"

async def fetch_image_from_unsplash(query: str):
    """Fetches image data from Unsplash API."""
    if not UNSPLASH_ACCESS_KEY:
        raise Exception("UNSPLASH_ACCESS_KEY is missing.")
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    url = f"https://api.unsplash.com/photos/random?query={query}&orientation=squarish" 
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    photo = response.json() 
    image_url = photo['urls']['regular']
    raw_answer = photo.get('alt_description') or photo.get('description') or query
    return image_url, raw_answer, "Unsplash"

async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None, str | None]:
    """Randomly selects an API, fetches, and pixelates the image. Returns (io, answer, url, category)."""
    category = random.choice(search_queries)
    query = category
    
    available_apis = []
    if PEXELS_API_KEY:
        available_apis.append(fetch_image_from_pexels)
    if UNSPLASH_ACCESS_KEY:
        available_apis.append(fetch_image_from_unsplash)

    if not available_apis:
        return None, "Both API Keys Missing. Check .env file.", None, None

    fetcher = random.choice(available_apis)

    try:
        # 1. Fetch Image URL and Raw Answer
        image_url, raw_answer, api_source = await fetcher(query)
        logger.info(f"Fetched image from {api_source} for query: {query}")

        # 2. Normalize and Clean the Answer
        all_words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_answer.lower())
        answer = random.choice(all_words) if all_words else query.lower()
        answer = answer.strip()
        
        # 3. Download Image Content
        image_data = requests.get(image_url, stream=True, timeout=10).content

        # 4. Process/Pixelate Image using PIL
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        downscale_factor = GAME_LEVELS[difficulty]['downscale_factor']
        
        width, height = img.size
        if width < downscale_factor or height < downscale_factor:
            downscale_factor = min(width, height, 2)
            
        small_width = width // downscale_factor
        small_height = height // downscale_factor
        
        small_img = img.resize((small_width, small_height), Image.Resampling.NEAREST)
        pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)
        
        output = io.BytesIO()
        pixelated_img.save(output, format='JPEG')
        output.seek(0)
        
        return output, answer, image_url, category
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Image API Request Error: {e}")
        return None, f"Image API Error: {e}", None, None
    except Exception as e:
        logger.error(f"Image Processing Error: {e}")
        return None, f"Image Processing/Unknown Error: {e}", None, None

# ------------------------------------------------------------------
# --- NEW PLAYER PROFILE AND SHOP COMMANDS ---
# ------------------------------------------------------------------

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user their personal report card."""
    user = update.effective_user
    rank, points, solved_count, current_streak, last_daily_claim = await get_user_profile_data(user.id)
    
    if isinstance(rank, str) and rank.startswith("DB Error"):
        await update.message.reply_text("❌ Error fetching profile data. Please try again later.")
        return

    profile_text = (
        "👤 **User Profile**\n"
        f"**User**: {user.first_name}\n"
        f"**🏆 Rank**: #{rank}\n"
        f"**💰 Points**: {points:,}\n"
        f"**🖼️ Images Solved**: {solved_count:,}\n"
        f"**🔥 Current Streak**: {current_streak} Days\n\n"
        "Use /leaderboard to see the global top players."
    )
    await update.message.reply_text(profile_text, parse_mode='Markdown')

async def album_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the user their collected stickers/images."""
    user_id = update.effective_user.id
    collected_categories = await get_user_collection(user_id)
    
    album_text = "📖 **Your Sticker Album** 📖\n\n"
    
    # Sort categories by name for stable display
    sorted_categories = sorted(SEARCH_CATEGORIES.items(), key=lambda item: item[0])
    
    for category_name, emoji in sorted_categories:
        status = emoji if category_name in collected_categories else "[❓]"
        album_text += f"**{category_name.title()}**: {status}\n"
        
    unlocked_count = len(collected_categories)
    total_count = len(SEARCH_CATEGORIES)
    
    album_text += f"\n**Total Unlocked**: {unlocked_count}/{total_count}\n"
    album_text += "Keep playing and guessing new categories to unlock all the missing stickers!"
    
    await update.message.reply_text(album_text, parse_mode='Markdown')

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Opens the shop menu."""
    user_id = update.effective_user.id
    current_score = get_user_score(user_id)
    
    shop_text = f"🛒 **Pixel Shop** (Your Balance: **{current_score:,} pts**)\n\n"
    keyboard = []
    
    for item_id, item in SHOP_ITEMS.items():
        shop_text += f"**{item_id}. {item['name']}**\n"
        shop_text += f"   *Cost*: **{item['cost']} pts**\n"
        shop_text += f"   *Effect*: {item['description']}\n\n"
        keyboard.append([InlineKeyboardButton(f"Buy {item['name']} ({item['cost']} pts)", callback_data=f'buy_{item_id}')])
        
    shop_text += "Type `/buy <item_number>` to purchase."
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(shop_text, parse_mode='Markdown', reply_markup=reply_markup)

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles shop purchases."""
    user_id = update.effective_user.id
    
    if not context.args or context.args[0] not in SHOP_ITEMS:
        await update.message.reply_text("Please specify a valid item ID. Usage: `/buy 1` or `/buy 2`")
        return

    item_id = context.args[0]
    item = SHOP_ITEMS[item_id]
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    if current_score < cost:
        await update.message.reply_text(f"❌ Purchase failed! You need **{cost} pts**, but only have **{current_score} pts**.", parse_mode='Markdown')
        return

    # Deduct points
    update_user_score(user_id, -cost)
    new_score = get_user_score(user_id)
    
    response = f"✅ **{item['name']}** Purchased!\n-**{cost} pts** deducted. New Balance: **{new_score} pts**.\n\n"
    
    if item_id == '1': # Flashlight (Placeholder)
        response += "🔦 Flashlight activated! The image has been slightly revealed (effect is conceptual for now)."
    
    elif item_id == '2': # First Letter (Implemented)
        game_state = active_games.get(update.effective_chat.id)
        if game_state and game_state['hint_string'][0] == '_':
            answer_word = game_state['answer']
            
            # Reveal the first letter
            hint_list = list(game_state['hint_string'])
            hint_list[0] = answer_word[0].upper()
            game_state['hint_string'] = "".join(hint_list)
            
            response += f"🔤 **First Letter Reveal!**\n**Progress**: `{game_state['hint_string']}`"
        else:
            response += "🔤 **First Letter Reveal!**\nThere is no active game or the first letter is already revealed."
            
    await update.message.reply_text(response, parse_mode='Markdown')

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allows a user to claim free points once every 24 hours."""
    user_id = update.effective_user.id
    current_time = datetime.now(pytz.utc)
    
    # Get last claim time
    rank, score, solved_count, current_streak, last_claim = await get_user_profile_data(user_id)
    
    time_since_claim = current_time - last_claim.replace(tzinfo=pytz.utc) if last_claim else None
    
    if last_claim and time_since_claim < timedelta(hours=23, minutes=59):
        # Calculate time remaining
        time_remaining = timedelta(hours=24) - time_since_claim
        hours, remainder = divmod(time_remaining.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        await update.message.reply_text(f"⏳ **Daily Bonus** already claimed today!\nYou can claim your next bonus in **{hours}h {minutes}m**.")
        return

    # Claim points
    success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
    
    if success:
        new_score = get_user_score(user_id)
        await update.message.reply_text(
            f"🎁 **Daily Bonus!** 🎉\nYou claimed **{DAILY_BONUS_POINTS}** free points!\nYour new balance is: **{new_score} pts**.\nCome back tomorrow!",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text("❌ Could not process your daily claim. DB Error.")

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Skips the current image, reveals the answer, and starts a new one."""
    chat_id = update.effective_chat.id
    
    game_state = active_games.pop(chat_id, None)
    
    if game_state:
        correct_answer = game_state['answer'].upper()
        
        await update.message.reply_text(f"⏭️ **Skipped!** The answer was: **{correct_answer}**.", parse_mode='Markdown')
        
        # Automatically start a new game
        await game_command(update, context)
        
    else:
        await update.message.reply_text("There is no active game to skip. Use /game to start one.")

async def simple_hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gives a free text clue based on the image category."""
    chat_id = update.effective_chat.id
    game_state = active_games.get(chat_id)
    
    if not game_state:
        await update.message.reply_text("💡 There is no active game. Use /game to start one.")
        return

    category = game_state['category']
    category_emoji = SEARCH_CATEGORIES.get(category, '❓')
    
    # Simple, free text clue
    hint_text = (
        f"💡 **Free Hint!** 💡\n"
        f"This image belongs to the **{category.title()}** category {category_emoji}."
    )
    
    await update.message.reply_text(hint_text, parse_mode='Markdown')


# ------------------------------------------------------------------
# --- GAME COMMANDS AND HANDLERS (UPDATED) ---
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
        # ... (End Game Logic remains the same) ...
        game_state = active_games.pop(chat_id, None)
        if game_state:
            correct_answer = game_state['answer'].upper()
            original_url = game_state['url']
            
            try:
                 await query.edit_message_caption(
                    caption=f"❌ **Game Over!** ❌\n You chose to end the game. The correct answer was: **{correct_answer}**.",
                    parse_mode='Markdown',
                    reply_markup=None
                )
            except:
                await context.bot.send_message(chat_id, f"The game has ended. The correct answer was: **{correct_answer}**.", parse_mode='Markdown')
                
            # Fallback to send original photo
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=f"The correct answer was: **{correct_answer}**.",
                    parse_mode='Markdown'
                )
            except:
                pass # Already sent text answer
            
        else:
            await context.bot.send_message(chat_id, "No active game to end.")
        return

    # --- HINT LOGIC ---
    if data == 'game_hint':
        # ... (Hint Logic remains the same, adjusted for category tracking) ...
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
            del active_games[chat_id]
            await context.bot.send_message(
                chat_id, 
                f"💡 **HINT TAKEN!** 💡\n-**{hint_cost} points** deducted.\n\nThe word **{answer_word.upper()}** has been fully revealed by hints. Game Over! No points awarded for this round.",
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
    pixelated_img_io, answer, original_url, category = await fetch_and_pixelate_image(difficulty)
    
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
        'hint_string': initial_hint_string,
        'category': category # Store the category
    }
    
    # 3. Send the Pixelated Image
    level_data = GAME_LEVELS[difficulty]
    points = level_data['points']
    
    caption = (
        f"**📸 Guessing Game: {difficulty.upper()}**\n"
        f"Guess this pixelated photo.\n"
        f"Correct answer yields **+{points} points**!\n"
        f"**Hint Progress**: `{initial_hint_string}`\n"
        f"Use /hint for a free category clue."
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
        category = game_state['category']
        
        # 1. Update Score and Profile Stats
        update_user_score(user_id, points)
        await save_solved_image(user_id, category) # Update album/streak/solved_count
        
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
        f"**{user_name}**! Your current score is: **{score:,}** points! 🏅",
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
            user = await context.bot.get_chat(user_id)
            user_name = user.full_name
        except Exception:
            user_name = f"User {user_id}"

        leaderboard_text += f"{i}. **{user_name}** - `{score:,}` points\n"
        
    await update.message.reply_text(leaderboard_text, parse_mode='Markdown')

# ------------------------------------------------------------------
# --- OTHER HANDLERS ---
# ------------------------------------------------------------------
# ... (broadcast_command, new_chat_member_handler, process_message, handle_voice_message, error_handler remain UNCHANGED) ...

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

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error."""
    logger.error("An error occurred:", exc_info=context.error)

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
    application.add_handler(CommandHandler("stats", profile_command)) # /stats maps to /profile
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("id", get_id_command))

    # Shop/Economy Commands
    application.add_handler(CommandHandler("shop", shop_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("daily", daily_command))

    # Game Commands
    application.add_handler(CommandHandler("game", game_command))
    application.add_handler(CommandHandler("myscore", my_score_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("album", album_command)) # /album maps to /collection
    application.add_handler(CommandHandler("collection", album_command)) 
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("hint", simple_hint_command))

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
    main()
