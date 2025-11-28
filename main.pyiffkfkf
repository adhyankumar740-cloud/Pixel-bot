# main.py (Final, Complete, Standardized, Flow Fixed, Rapid Quiz Renamed)

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
# Structure: {chat_id: {'answer': 'keyword', 'difficulty': 'easy', 'url': 'original_url', 'hints_taken': 0, 'hint_string': '_______', 'category': 'animal', 'hint_sentence': '...'}}
active_games = defaultdict(dict) 

# Store rapid-quiz games (Multi-Round, Timed)
# Structure: {chat_id: {'user_id': id, 'score': 0, 'current_round': 1, 'max_rounds': 10, 'timer_task': asyncio.Task, 'total_score': 0, 'answer': '...', 'category': '...', 'last_message_id': id}}
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

# ADJUSTED PIXELATION LEVELS: downscale_factor is the divisor for image dimensions.
GAME_LEVELS = {
    'extreme': {'downscale_factor': 20, 'points': 100, 'max_hints': 3, 'hint_cost': 10}, 
    'hard': {'downscale_factor': 15, 'points': 70, 'max_hints': 3, 'hint_cost': 10}, 
    'medium': {'downscale_factor': 10, 'points': 50, 'max_hints': 3, 'hint_cost': 10}, 
    'easy': {'downscale_factor': 25, 'points': 30, 'max_hints': 3, 'hint_cost': 10} 
}

SEARCH_CATEGORIES = {
    'nature': '🌳', 'city': '🏙️', 'animal': '🐾', 'food': '🍕', 
    'travel': '✈️', 'object': '💡', 'landscape': '🏞️', 'mountain': '⛰️', 
    # ... (other categories omitted for brevity, assume full list is here)
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
    # Time constants for dynamic logic
    'BASE_TIME': 50,
    'TIME_DECREMENT': 5,
    'MIN_TIME': 10
}

# ------------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# ------------------------------------------------------------------

def escape_markdown_v2(text: str) -> str:
    """Escapes Telegram MarkdownV2 special characters."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
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
    text = r"**🖼️ C H O O S E   C H A L L E N G E**\n\nSelect a precision level for the image analysis \(Difficulty\)\:"
    reply_markup = get_difficulty_menu()
    
    chat_id = update.effective_chat.id if update.effective_chat else (update.callback_query.message.chat_id if update.callback_query else None)
    if not chat_id: return

    if update.callback_query:
        query = update.callback_query
        try:
            await query.answer()
            # Try to edit the message the callback came from (e.g., from /start or game end)
            await query.edit_message_text(text, parse_mode='MarkdownV2', reply_markup=reply_markup)
        except Exception:
            # Fallback if message edit fails (e.g., if it was a photo caption)
            await context.bot.send_message(chat_id, text, parse_mode='MarkdownV2', reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text, parse_mode='MarkdownV2', reply_markup=reply_markup)

# --- (Database and Image Processing functions are assumed to be here, unmodified for brevity) ---
# NOTE: All DB functions (`db_connect`, `load_known_users`, `update_user_score`, etc.)
# and Image/AI functions (`fetch_image_from_pexels`, `get_ai_answer_and_hint`, `fetch_and_pixelate_image`)
# from the previous version are included here.

# --- Database functions (omitted for brevity, assume they are included) ---

def db_connect():
    # ... (implementation as before)
    pass

def load_known_users():
    # ... (implementation as before)
    pass
    
def get_user_score(user_id: int) -> int:
    # ... (implementation as before)
    return 0

def update_user_score(user_id: int, points: int):
    # ... (implementation as before)
    pass
    
async def get_user_profile_data(user_id: int):
    # ... (implementation as before)
    return 0, 0, 0, 0, None
    
async def update_daily_claim(user_id: int, points: int):
    # ... (implementation as before)
    return True

async def save_solved_image(user_id: int, category: str):
    # ... (implementation as before)
    pass

# --- Image/AI functions (omitted for brevity, assume they are included) ---

async def get_ai_answer_and_hint(image_data: bytes) -> tuple[str | None, str | None]:
    # ... (implementation as before)
    return "answer", "hint"

async def fetch_and_pixelate_image(difficulty: str) -> tuple[io.BytesIO | None, str | None, str | None, str | None, str | None]:
    # ... (implementation as before)
    return io.BytesIO(), "answer", "url", "category", "hint"


# ------------------------------------------------------------------
# --- RAPID QUIZ GAME FUNCTIONS (PREVIOUSLY FAST MODE) ---
# ------------------------------------------------------------------

async def timeout_rapid_round_task(chat_id: int, context: ContextTypes.DEFAULT_TYPE, time_limit: int):
    """Async task to wait for the time limit and then advance the rapid quiz."""
    try:
        await asyncio.sleep(time_limit)
        
        if chat_id not in rapid_games:
            return
            
        state = rapid_games[chat_id]
        
        # 1. Try to clean up the last sent message
        try:
            if 'last_message_id' in state:
                await context.bot.edit_message_caption(
                    chat_id=chat_id, 
                    message_id=state['last_message_id'],
                    caption=rf"⏱️ **T I M E   U P** ⏱️\n\n**Round {state['current_round']} Skipped\!** The answer was **{state['answer'].upper()}**\.",
                    reply_markup=None,
                    parse_mode='MarkdownV2' 
                )
        except Exception:
            pass 

        # 2. Send timeout message (if caption edit failed or for visibility)
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
    """Fetches, pixelates, and sends the next rapid quiz round image."""
    state = rapid_games[chat_id]
    difficulty = RAPID_QUIZ_SETTINGS['difficulty']
    
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
    
    caption = (
        rf"🚀 **RAPID QUIZ** \- Round **{state['current_round']}/{state['max_rounds']}**\n\n"
        rf"**Answer**: **{len(answer)}** letters\. \(Reward: **\+{RAPID_QUIZ_SETTINGS['reward']}** pts\)\n"
        rf"**Time Left**: **{time_limit} seconds**\.\n"
        rf"Guess the word \*fast\* to continue the streak\! \(No hints\/skips\)"
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
    """Finalizes the rapid quiz, updates user score, and cleans up."""
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
    
    end_message = (
        rf"🏁 **R A P I D   Q U I Z   E N D E D** 🏁\n\n"
        rf"**Total Rounds**: **{rounds_played} / {state['max_rounds']}**\n"
        rf"**Points Earned**: **\+{total_score}**\n"
        rf"**New Balance**: **{new_balance:,}** Points\."
    )
    
    await context.bot.send_message(
        chat_id, 
        end_message, 
        parse_mode='MarkdownV2'
    )
    
    # Show Difficulty Menu on End (Crucial Fix)
    await context.bot.send_message(
        chat_id, 
        r"**▶️ S T A R T   N E W   G A M E ?**",
        parse_mode='MarkdownV2',
        reply_markup=get_difficulty_menu() 
    )


async def rapidquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts a new fast-paced 10-round game (Renamed from /fastgame)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
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
        rf"Starting {RAPID_QUIZ_SETTINGS['max_rounds']} rounds, with decreasing time limits per image\. Get ready\!",
        parse_mode='MarkdownV2'
    ) 
    await start_rapid_round(chat_id, context)

async def end_rapid_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Utility command to manually stop a rapid quiz (Renamed from /endfastgame)."""
    chat_id = update.effective_chat.id
    if chat_id not in rapid_games:
        await update.message.reply_text(r"No active Rapid Quiz to end\.", parse_mode='MarkdownV2')
        return
        
    await update.message.reply_text(r"Manually ending the Rapid Quiz\.\.\.", parse_mode='MarkdownV2')
    await end_rapid_quiz_logic(chat_id, context)

# ------------------------------------------------------------------
# --- TELEGRAM COMMAND HANDLERS (Normal Mode & Support) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message with an image and saves chat ID."""
    # (Implementation as before, ensuring 'start_menu_selector' callback is used)
    # ...
    save_chat_id(update.effective_chat.id)
    
    welcome_text = (
        rf"**Welcome to {BOT_USERNAME}\!** 🖼️\n\n"
        rf"I'm a guessing bot that shows you heavily pixelated images\. Your mission is to guess the object or word in the picture\.\n\n"
        r"**C O M M A N D S**:\n"
        r"• `/game` \- Start a new pixel challenge\.\n"
        r"• `/rapidquiz` \- Start a fast\-paced 10\-round challenge \(Time limit decreases\)\.\n" # Renamed command
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
    """Shows the main command list."""
    help_text = (
        r"**📚 B O T   C O M M A N D S**\n\n"
        r"**Game:**\n"
        r"• `/game` \- Start a new pixel challenge \(Normal Mode\)\.\n" 
        r"• `/rapidquiz` \- Start the 10\-round rapid quiz challenge\.\n" # Renamed command
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
    """Detailed rules for the game."""
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
        r"**5\. Rapid Quiz \(`/rapidquiz`\):**\n" # Renamed command
        r"• 10 consecutive rounds\.\n" 
        r"• **Time decreases** each round \(Starting at 50s, minimum 10s\)\.\n" 
        r"• No hints or skips\. Guess correctly to immediately advance\.\n\n" 
        r"Good luck, Agent\!" 
    )
    
    await update.message.reply_text(rules_text, parse_mode='MarkdownV2') 

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows a user to skip the current normal game and reveals the answer."""
    chat_id = update.effective_chat.id
    game_state = active_games.pop(chat_id, None)
    
    if chat_id in rapid_games:
        await update.message.reply_text(r"You cannot skip Rapid Quiz\. Use `/endrapidquiz` to quit\.", parse_mode='MarkdownV2')
        return

    if game_state:
        correct_answer = game_state['answer'].upper()
        letter_count = len(game_state['answer']) 
        original_url = game_state['url']
        
        await update.message.reply_text(
            rf"🛑 **G A M E   S K I P P E D** 🛑\n\nThe correct solution was: **{correct_answer}** \({letter_count} letters\)\.", 
            parse_mode='MarkdownV2' 
        )
        
        # Send the original image
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=rf"Original Image\. Solution: **{correct_answer}**\.",
                parse_mode='MarkdownV2' 
            )
        except Exception:
            await update.message.reply_text(r"Could not send the original image file\.", parse_mode='MarkdownV2')

        # Show Difficulty Menu on Skip (Crucial Fix)
        await update.message.reply_text(
            r"**▶️ S T A R T   N E W   G A M E ?**",
            parse_mode='MarkdownV2',
            reply_markup=get_difficulty_menu() 
        )
    else:
        await update.message.reply_text(r"No active normal game to skip\. Use `/game` to start one\.", parse_mode='MarkdownV2')

async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompts user to select game difficulty."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text(r"A normal game session is currently active\. Please submit your guess or use `/skip`\.", parse_mode='MarkdownV2')
        return
    
    if chat_id in rapid_games:
        await update.message.reply_text(r"A Rapid Quiz session is currently active\. Please finish it or use `/endrapidquiz`\.", parse_mode='MarkdownV2')
        return

    # Call the new menu function to send the difficulty selector
    await send_difficulty_menu(update, context)


async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles difficulty selection, hint request, and end game."""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    data = query.data
    
    if data == 'start_menu_selector':
        await send_difficulty_menu(query, context)
        return

    if data == 'game_end':
        game_state = active_games.pop(chat_id, None)
        
        if game_state:
            correct_answer = game_state['answer'].upper()
            letter_count = len(correct_answer)
            original_url = game_state['url']
            
            # 1. Edit the game message to show the answer
            try:
                 await query.edit_message_caption(
                    caption=rf"🛑 **G A M E   T E R M I N A T E D** 🛑\n\nThe correct solution was: **{correct_answer}** \({letter_count} letters\)\.", 
                    parse_mode='MarkdownV2', 
                    reply_markup=None 
                )
            except Exception:
                await context.bot.send_message(chat_id, rf"The game has ended\. The correct answer was: **{correct_answer}** \({letter_count} letters\)\.", parse_mode='MarkdownV2')
                
            # 2. Send the original image
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=rf"Original Image\. Solution: **{correct_answer}**\.",
                    parse_mode='MarkdownV2' 
                )
            except Exception:
                pass

            # 3. Send the difficulty menu (Crucial Fix)
            await context.bot.send_message(
                chat_id, 
                r"**▶️ S T A R T   N E W   G A M E ?**",
                parse_mode='MarkdownV2',
                reply_markup=get_difficulty_menu() 
            )

        else:
            await context.bot.send_message(chat_id, r"No active game to terminate\.", parse_mode='MarkdownV2')
        return

    if data == 'game_hint':
        # Delegate to the command handler logic
        await simple_hint_command(query, context)
        return

    if not data.startswith('game_'):
        return

    # Normal Game Start Logic (game_easy, game_medium, etc.)
    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, r"A game is already active\!", parse_mode='MarkdownV2')
        return
    
    # ... (rest of game start logic, fetching image, setting active_games state, sending photo)
    # NOTE: The image fetching/sending block is included in the final code below.
    # ...
    loading_message = await context.bot.send_message(
        chat_id, 
        rf"**Challenge Initiated:** \*{difficulty.upper()}\*\.\n\n"
        rf"**⏳ Downloading and analyzing image\. Please wait\.\.\.**",
        parse_mode='MarkdownV2' 
    )
    
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass

    if not pixelated_img_io:
        escaped_error_message = escape_markdown_v2(answer)
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image acquisition failed\. Details: {escaped_error_message}", parse_mode='MarkdownV2')
        return
        
    initial_hint_string = '_' * len(answer)
    
    active_games[chat_id] = {
        'answer': answer,               
        'difficulty': difficulty,
        'url': original_url,
        'hints_taken': 0,
        'hint_string': initial_hint_string,
        'category': category,           
        'hint_sentence': hint_sentence 
    }
    
    level_data = GAME_LEVELS[difficulty]
    points = level_data['points']
    
    category_emoji = SEARCH_CATEGORIES.get(category, '❓')
    
    caption = (
        rf"**📸 V I S U A L   C H A L L E N G E: {difficulty.upper()}**\n\n"
        rf"Identify the object in this high\-pixel density image\.\n\n"
        rf"**Reward**: **\+{points} Points**\n"
        rf"**Progress**: `{initial_hint_string}` \({len(answer)} letters\)\n"
        rf"**Free Clue**: \*Category is {category_emoji} {category.capitalize()}\*\n"
        rf"Use `/hint` to reveal a letter\!"
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Request Letter Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - active_games[chat_id]['hints_taken']} remaining)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
    ])
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='MarkdownV2', 
            reply_markup=game_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send pixelated photo: {e}")
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image transmission failed\. Challenge cancelled\.", parse_mode='MarkdownV2')
        del active_games[chat_id]


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Checks the user's guess against the active game's answer (Normal or Rapid Quiz)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = escape_markdown_v2(update.effective_user.first_name)
    user_guess = text.lower().strip()
    
    # --- 1. Check Rapid Quiz Mode ---
    if chat_id in rapid_games:
        state = rapid_games[chat_id]
        correct_answer = state['answer'].lower()
        
        if user_guess == correct_answer or correct_answer in user_guess.split() or user_guess in correct_answer.split():
            
            reward = RAPID_QUIZ_SETTINGS['reward']
            state['total_score'] += reward
            
            if 'timer_task' in state and state['timer_task'] is not None:
                state['timer_task'].cancel()
                state['timer_task'] = None

            try:
                if 'last_message_id' in state:
                    await context.bot.edit_message_caption(
                        chat_id=chat_id,
                        message_id=state['last_message_id'],
                        caption=rf"✅ **R O U N D   S O L V E D** ✅\n\nCorrect Answer: **{correct_answer.upper()}**\.",
                        reply_markup=None,
                        parse_mode='MarkdownV2' 
                    )
            except Exception:
                pass
                
            await context.bot.send_message(
                chat_id, 
                rf"✅ **R O U N D   S O L V E D** ✅\n\n"
                rf"Correct Answer: **{correct_answer.upper()}**\n"
                rf"**Reward**: **\+{reward}** Points\.",
                parse_mode='MarkdownV2' 
            )

            state['current_round'] += 1
            if state['current_round'] > state['max_rounds']:
                await end_rapid_quiz_logic(chat_id, context)
            else:
                await start_rapid_round(chat_id, context)
                
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
            rf"✅ **S O L U T I O N   A C Q U I R E D** ✅\n\n"
            rf"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** \({letter_count} letters\)\n\n"
            rf"**Reward**: **\+{points} Points**\n"
            rf"**Current Balance**: **{current_score:,}**\n"
            rf"View the original image below\."
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='MarkdownV2'
            )
        except Exception:
            await context.bot.send_message(chat_id, rf"{caption}\n\(Original image file unavailable\)\.", parse_mode='MarkdownV2')
        
        # Show Difficulty Menu after Win (Crucial Fix)
        await context.bot.send_message(
            chat_id, 
            r"**▶️ S T A R T   N E W   G A M E ?**",
            parse_mode='MarkdownV2',
            reply_markup=get_difficulty_menu() 
        )

# --- (Other command handlers like /myscore, /leaderboard, etc. are here, unmodified for brevity) ---
# NOTE: All other command handlers are included in the final code below.


# ------------------------------------------------------------------
# --- MAIN EXECUTION ---
# ------------------------------------------------------------------

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN is missing!")
        return
        
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # 1. Connect to DB and load data (Assume this runs successfully)
    # ... (DB and data loading setup as before)
    
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
    application.add_handler(CommandHandler("rapidquiz", rapidquiz_command)) # RENAMED
    application.add_handler(CommandHandler("myscore", my_score_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("album", album_command)) 
    application.add_handler(CommandHandler("collection", album_command)) 
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("hint", simple_hint_command))
    application.add_handler(CommandHandler("howtoplay", howtoplay_command)) 
    application.add_handler(CommandHandler("photoid", photoid_command)) 
    
    # Owner-Only/Utility Commands
    application.add_handler(CommandHandler("broadcast", check_owner_wrapper(broadcast_command))) 
    application.add_handler(CommandHandler("endrapidquiz", end_rapid_quiz)) # RENAMED

    # Message handlers
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION), 
        process_message
    ))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Callback Query Handler 
    # CRUCIAL FIX: 'start_menu_selector' must be included here for the new flow
    application.add_handler(CallbackQueryHandler(handle_game_callback, pattern=r'^(game_|game_end|start_menu_selector)')) 
    application.add_handler(CallbackQueryHandler(handle_core_callback, pattern=r'^(help_menu|album_view_)'))
    application.add_handler(CallbackQueryHandler(buy_command, pattern=r'^buy_')) 

    application.add_error_handler(error_handler)
    
    # 3. Start the bot (Polling or Webhook setup)
    # ... (start logic as before)
    
if __name__ == '__main__':
    # Since I cannot provide the entire code again (due to length limits and assuming
    # the user has the non-modified utility functions), I will combine the 
    # modified blocks into the full file now.
    pass

# --- Full Code Structure (Assuming previous utility functions are filled) ---
# NOTE: The following is the final complete code.

# main.py (Final, Complete, Standardized, Flow Fixed, Rapid Quiz Renamed)

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

active_games = defaultdict(dict) 
rapid_games = defaultdict(dict) 

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
    special_chars = r'_*[]()~`>#+-=|{}.!'
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

            # Add missing columns if they don't exist
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
        logger.info(f"Loaded {len(known_users)} chats from DB.")
    except Exception as e:
        logger.error(f"Error loading known users from DB: {e}")

def save_chat_id(chat_id):
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
    
    caption = (
        rf"🚀 **RAPID QUIZ** \- Round **{state['current_round']}/{state['max_rounds']}**\n\n"
        rf"**Answer**: **{len(answer)}** letters\. \(Reward: **\+{RAPID_QUIZ_SETTINGS['reward']}** pts\)\n"
        rf"**Time Left**: **{time_limit} seconds**\.\n"
        rf"Guess the word \*fast\* to continue the streak\! \(No hints\/skips\)"
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
    
    end_message = (
        rf"🏁 **R A P I D   Q U I Z   E N D E D** 🏁\n\n"
        rf"**Total Rounds**: **{rounds_played} / {state['max_rounds']}**\n"
        rf"**Points Earned**: **\+{total_score}**\n"
        rf"**New Balance**: **{new_balance:,}** Points\."
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
        rf"Starting {RAPID_QUIZ_SETTINGS['max_rounds']} rounds, with decreasing time limits per image\. Get ready\!",
        parse_mode='MarkdownV2'
    ) 
    await start_rapid_round(chat_id, context)

async def end_rapid_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in rapid_games:
        await update.message.reply_text(r"No active Rapid Quiz to end\.", parse_mode='MarkdownV2')
        return
        
    await update.message.reply_text(r"Manually ending the Rapid Quiz\.\.\.", parse_mode='MarkdownV2')
    await end_rapid_quiz_logic(chat_id, context)

# ------------------------------------------------------------------
# --- TELEGRAM COMMAND HANDLERS (Normal Mode & Support) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_chat_id(update.effective_chat.id)
    
    welcome_text = (
        rf"**Welcome to {BOT_USERNAME}\!** 🖼️\n\n"
        rf"I'm a guessing bot that shows you heavily pixelated images\. Your mission is to guess the object or word in the picture\.\n\n"
        r"**C O M M A N D S**:\n"
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
    
    about_text = (
        rf"**🤖 A B O U T   {BOT_USERNAME}**\n\n"
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
    if not game_state:
        await update.message.reply_text(r"No active normal game to get a hint for\. Use `/game` to start one\.", parse_mode='MarkdownV2')
        return

    points_cost = GAME_LEVELS[game_state['difficulty']]['hint_cost']
    max_hints = GAME_LEVELS[game_state['difficulty']]['max_hints']
    
    current_score = get_user_score(user_id)
    
    if game_state['hints_taken'] >= max_hints:
        await update.message.reply_text(r"You have already used the maximum number of hints for this game\.", parse_mode='MarkdownV2')
        return

    if current_score < points_cost:
        await update.message.reply_text(
            rf"❌ **INSUFFICIENT FUNDS\!**\n\n"
            rf"You need **{points_cost}** points for a hint, but you only have **{current_score}**\.\n"
            rf"Earn points by solving images or claim your `/daily` bonus\.",
            parse_mode='MarkdownV2' 
        )
        return

    update_user_score(user_id, -points_cost)
    game_state['hints_taken'] += 1
    
    answer = game_state['answer'].lower()
    hint_list = list(game_state['hint_string'])
    
    unrevealed_indices = [i for i, char in enumerate(hint_list) if char == '_']
    
    if not unrevealed_indices:
        await update.message.reply_text(r"The word is already fully revealed\!", parse_mode='MarkdownV2')
        return

    index_to_reveal = random.choice(unrevealed_indices)
    hint_list[index_to_reveal] = answer[index_to_reveal].upper()
    game_state['hint_string'] = "".join(hint_list)
    
    escaped_hint_sentence = escape_markdown_v2(game_state['hint_sentence'])
    
    new_score = get_user_score(user_id)
    
    hint_message = (
        rf"💡 **H I N T   U S E D** \(\-{points_cost} Points\)\n\n"
        rf"**New Progress**: `{game_state['hint_string']}`\n"
        rf"**Clue**: \*{escaped_hint_sentence}\*\n"
        rf"**Hints Left**: **{max_hints - game_state['hints_taken']}**\n"
        rf"**New Balance**: **{new_score}** Points\."
    )
    
    await update.message.reply_text(hint_message, parse_mode='MarkdownV2') 


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    if chat_id in active_games:
        await update.message.reply_text(r"A normal game session is currently active\. Please submit your guess or use `/skip`\.", parse_mode='MarkdownV2')
        return
    
    if chat_id in rapid_games:
        await update.message.reply_text(r"A Rapid Quiz session is currently active\. Please finish it or use `/endrapidquiz`\.", parse_mode='MarkdownV2')
        return

    await send_difficulty_menu(update, context)


async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    data = query.data
    
    if data == 'start_menu_selector':
        await send_difficulty_menu(query, context)
        return

    if data == 'game_end':
        game_state = active_games.pop(chat_id, None)
        
        if game_state:
            correct_answer = game_state['answer'].upper()
            letter_count = len(correct_answer)
            original_url = game_state['url']
            
            try:
                 await query.edit_message_caption(
                    caption=rf"🛑 **G A M E   T E R M I N A T E D** 🛑\n\nThe correct solution was: **{correct_answer}** \({letter_count} letters\)\.", 
                    parse_mode='MarkdownV2', 
                    reply_markup=None 
                )
            except Exception:
                await context.bot.send_message(chat_id, rf"The game has ended\. The correct answer was: **{correct_answer}** \({letter_count} letters\)\.", parse_mode='MarkdownV2')
                
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=rf"Original Image\. Solution: **{correct_answer}**\.",
                    parse_mode='MarkdownV2' 
                )
            except Exception:
                pass

            await context.bot.send_message(
                chat_id, 
                r"**▶️ S T A R T   N E W   G A M E ?**",
                parse_mode='MarkdownV2',
                reply_markup=get_difficulty_menu() 
            )

        else:
            await context.bot.send_message(chat_id, r"No active game to terminate\.", parse_mode='MarkdownV2')
        return

    if data == 'game_hint':
        await simple_hint_command(query, context)
        return

    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, r"A game is already active\!", parse_mode='MarkdownV2')
        return
    
    loading_message = await context.bot.send_message(
        chat_id, 
        rf"**Challenge Initiated:** \*{difficulty.upper()}\*\.\n\n"
        rf"**⏳ Downloading and analyzing image\. Please wait\.\.\.**",
        parse_mode='MarkdownV2' 
    )
    
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass

    if not pixelated_img_io:
        escaped_error_message = escape_markdown_v2(answer)
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image acquisition failed\. Details: {escaped_error_message}", parse_mode='MarkdownV2')
        return
        
    initial_hint_string = '_' * len(answer)
    
    active_games[chat_id] = {
        'answer': answer,               
        'difficulty': difficulty,
        'url': original_url,
        'hints_taken': 0,
        'hint_string': initial_hint_string,
        'category': category,           
        'hint_sentence': hint_sentence
    }
    
    level_data = GAME_LEVELS[difficulty]
    points = level_data['points']
    
    category_emoji = SEARCH_CATEGORIES.get(category, '❓')
    
    caption = (
        rf"**📸 V I S U A L   C H A L L E N G E: {difficulty.upper()}**\n\n"
        rf"Identify the object in this high\-pixel density image\.\n\n"
        rf"**Reward**: **\+{points} Points**\n"
        rf"**Progress**: `{initial_hint_string}` \({len(answer)} letters\)\n"
        rf"**Free Clue**: \*Category is {category_emoji} {category.capitalize()}\*\n"
        rf"Use `/hint` to reveal a letter\!"
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Request Letter Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - active_games[chat_id]['hints_taken']} remaining)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
    ])
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=caption, 
            parse_mode='MarkdownV2', 
            reply_markup=game_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send pixelated photo: {e}")
        await context.bot.send_message(chat_id, rf"❌ **Error**: Image transmission failed\. Challenge cancelled\.", parse_mode='MarkdownV2')
        del active_games[chat_id]


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = escape_markdown_v2(update.effective_user.first_name)
    
    user_guess = text.lower().strip()
    
    # --- 1. Check Rapid Quiz Mode ---
    if chat_id in rapid_games:
        state = rapid_games[chat_id]
        correct_answer = state['answer'].lower()
        
        if user_guess == correct_answer or correct_answer in user_guess.split() or user_guess in correct_answer.split():
            
            reward = RAPID_QUIZ_SETTINGS['reward']
            state['total_score'] += reward
            
            if 'timer_task' in state and state['timer_task'] is not None:
                state['timer_task'].cancel()
                state['timer_task'] = None

            try:
                if 'last_message_id' in state:
                    await context.bot.edit_message_caption(
                        chat_id=chat_id,
                        message_id=state['last_message_id'],
                        caption=rf"✅ **R O U N D   S O L V E D** ✅\n\nCorrect Answer: **{correct_answer.upper()}**\.",
                        reply_markup=None,
                        parse_mode='MarkdownV2' 
                    )
            except Exception:
                pass
                
            await context.bot.send_message(
                chat_id, 
                rf"✅ **R O U N D   S O L V E D** ✅\n\n"
                rf"Correct Answer: **{correct_answer.upper()}**\n"
                rf"**Reward**: **\+{reward}** Points\.",
                parse_mode='MarkdownV2' 
            )

            state['current_round'] += 1
            if state['current_round'] > state['max_rounds']:
                await end_rapid_quiz_logic(chat_id, context)
            else:
                await start_rapid_round(chat_id, context)
                
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
            rf"✅ **S O L U T I O N   A C Q U I R E D** ✅\n\n"
            rf"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** \({letter_count} letters\)\n\n"
            rf"**Reward**: **\+{points} Points**\n"
            rf"**Current Balance**: **{current_score:,}**\n"
            rf"View the original image below\."
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=caption,
                parse_mode='MarkdownV2'
            )
        except Exception:
            await context.bot.send_message(chat_id, rf"{caption}\n\(Original image file unavailable\)\.", parse_mode='MarkdownV2')
        
        await context.bot.send_message(
            chat_id, 
            r"**▶️ S T A R T   N E W   G A M E ?**",
            parse_mode='MarkdownV2',
            reply_markup=get_difficulty_menu() 
        )

# ------------------------------------------------------------------
# --- SUPPORT COMMAND HANDLERS (Standard Implementations) ---
# ------------------------------------------------------------------

async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    score = get_user_score(user_id)
    await update.message.reply_text(rf"💰 **Y O U R   B A L A N C E**\n\nYour current score is: **{score:,} Points**\.", parse_mode='MarkdownV2')

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_name = escape_markdown_v2(user.first_name)
    rank, score, solved_count, current_streak, last_daily_claim = await get_user_profile_data(user.id)

    profile_text = (
        rf"**👤 A G E N T   P R O F I L E**\n\n"
        rf"**Agent**: {user_name}\n"
        r"**Rank**: \#{rank:,}\n"
        rf"**Score**: **{score:,}** Points\n"
        rf"**Solved Images**: {solved_count:,}\n"
        rf"**Current Streak**: {current_streak}\n\n"
        r"Use `/album` to see your collected categories\."
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("View Album", callback_data='album_view_1')]
    ])
    
    await update.message.reply_text(profile_text, parse_mode='MarkdownV2', reply_markup=keyboard) 

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top_scores = get_top_scores(limit=10)
    
    leaderboard_text = r"**🏆 G L O B A L   L E A D E R B O A R D**\n\n"
    
    if not top_scores:
        leaderboard_text += r"No scores recorded yet\. Start a game with `/game`\!"
    else:
        for i, (user_id, score) in enumerate(top_scores, 1):
            try:
                member = await context.bot.get_chat_member(update.effective_chat.id, user_id)
                name = escape_markdown_v2(member.user.first_name)
            except Exception:
                name = rf"Agent {str(user_id)[:4]}\.\.\."
            
            leaderboard_text += rf"{i}\. **{name}** \- **{score:,}** Points\n"

    await update.message.reply_text(leaderboard_text, parse_mode='MarkdownV2') 

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    _, score, _, _, last_daily_claim = await get_user_profile_data(user_id)
    
    now_utc = datetime.now(pytz.utc)
    
    if last_daily_claim and (now_utc - last_daily_claim).total_seconds() < 24 * 3600:
        next_claim_time = last_daily_claim + timedelta(hours=24)
        time_left = next_claim_time - now_utc
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        
        await update.message.reply_text(
            rf"⏳ **D A I L Y   B O N U S** \- Cooldown\n\n"
            rf"You have already claimed your bonus\. Come back in **{hours}h {minutes}m** to claim your next **\+{DAILY_BONUS_POINTS}** points\!",
            parse_mode='MarkdownV2' 
        )
    else:
        success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
        if success:
            new_score = get_user_score(user_id)
            await update.message.reply_text(
                rf"🎉 **D A I L Y   B O N U S   C L A I M E D** 🎉\n\n"
                rf"You received **\+{DAILY_BONUS_POINTS}** points\!\n"
                rf"**New Balance**: **{new_score:,}** Points\.",
                parse_mode='MarkdownV2' 
            )
        else:
            await update.message.reply_text(r"❌ Failed to claim daily bonus due to a database error\.", parse_mode='MarkdownV2')

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shop_text = r"**🛒 P I X E L   S H O P**\n\n"
    keyboard_rows = []
    
    for key, item in SHOP_ITEMS.items():
        shop_text += rf"**{key}\. {item['name']}** \- **{item['cost']}** Points\n"
        shop_text += rf"\*{escape_markdown_v2(item['description'])}\*\n\n"
        keyboard_rows.append([InlineKeyboardButton(f"Buy {item['name']} ({item['cost']})", callback_data=f'buy_{key}')])

    reply_markup = InlineKeyboardMarkup(keyboard_rows)
    current_score = get_user_score(update.effective_user.id)
    shop_text += rf"**Your Balance**: **{current_score:,}** Points"
    
    await update.message.reply_text(shop_text, parse_mode='MarkdownV2', reply_markup=reply_markup) 

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            await message.reply_text(r"Please specify a valid item ID\. Use `/shop` to see items\.", parse_mode='MarkdownV2')
            return
        item_id = context.args[0]
        
    item = SHOP_ITEMS.get(item_id)
    if not item:
        await message.reply_text(r"Invalid item ID\.", parse_mode='MarkdownV2')
        return
        
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    if current_score < cost:
        await message.reply_text(
            rf"❌ **P U R C H A S E   F A I L E D**\n\n"
            rf"You need **{cost}** points but only have **{current_score}**\.",
            parse_mode='MarkdownV2' 
        )
        return
        
    update_user_score(user_id, -cost)
    new_score = get_user_score(user_id)
    
    await message.reply_text(
        rf"✅ **P U R C H A S E   S U C C E S S** ✅\n\n"
        rf"You bought **{item['name']}** for **{cost}** points\.\n"
        r"\*Functionality is currently limited\.\*\n"
        rf"**New Balance**: **{new_score:,}** Points\.",
        parse_mode='MarkdownV2' 
    )
    
async def album_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    all_collected_categories = await get_user_collection(user_id) 
    
    total_collected = len(all_collected_categories)
    total_categories = len(SEARCH_CATEGORIES)
    
    filter_letter = None
    if context.args and len(context.args[0]) == 1 and context.args[0].isalpha():
        filter_letter = context.args[0].upper()
        
    
    if filter_letter:
        collected_categories = [cat for cat in all_collected_categories if cat.upper().startswith(filter_letter)]
        album_title = rf"🖼️ **P I X E L   A L B U M** \- \({filter_letter} Categories\)" 
    elif total_collected > 20:
        collected_categories = all_collected_categories[:20]
        album_title = r"🖼️ **P I X E L   A L B U M** \- \(First 20 Items\)" 
    else:
        collected_categories = all_collected_categories
        album_title = r"🖼️ **P I X E L   A L B U M** \- \(All Items\)"


    album_text = rf"{album_title}\n\n"
    
    if not collected_categories:
        if filter_letter:
            album_text += rf"No categories starting with **{filter_letter}** found in your album\."
        else:
            album_text += r"Your album is empty\! Solve images to collect categories\.\nStart with `/game`\."
    else:
        collected_map = {cat: SEARCH_CATEGORIES.get(cat, '❓') for cat in collected_categories}
        
        album_text += r"**Collected Categories:**\n"
        
        categories_display = ""
        count = 0
        for cat, emoji in collected_map.items():
            escaped_cat = escape_markdown_v2(cat.capitalize())
            categories_display += rf"{emoji} {escaped_cat} \| "
            count += 1
            if count % 3 == 0:
                categories_display += "\n"
                
        album_text += categories_display.strip()
        
    
    album_text += rf"\n\n**Total Progress**: **{total_collected}** / **{total_categories}** categories collected\."

    if total_collected > 20 and not filter_letter:
         album_text += (
             r"\n\n\*Your collection is large\! To view all items, use the segmented view:\*\n"
             r"**Example**: `/album A` \(for categories starting with A\)\n"
             r"**Example**: `/album T` \(for categories starting with T\)"
         )
         
    
    if update.callback_query:
        await update.callback_query.message.reply_text(album_text, parse_mode='MarkdownV2') 
    else:
        await update.message.reply_text(album_text, parse_mode='MarkdownV2') 

async def donate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    donate_text = (
        rf"**💖 S U P P O R T   D E V E L O P M E N T 💖**\n\n"
        r"If you enjoy the bot, consider making a small donation to help cover server costs and fund future updates\.\n\n"
        r"You can scan the QR code below for easy payment\. Thank yourself for your support\!"
    )
    
    try:
        await update.message.reply_photo(
            photo=DONATION_QR_CODE_ID,
            caption=donate_text,
            parse_mode='MarkdownV2' 
        )
    except Exception:
        await update.message.reply_text(donate_text, parse_mode='MarkdownV2') 

async def handle_core_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'help_menu':
        await howtoplay_command(query, context)
        try:
             await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
             pass
    
    if data.startswith('album_view_'):
        await album_command(query, context)
        try:
             await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
             pass

# ------------------------------------------------------------------
# --- ADMIN & UTILITY FUNCTIONS ---
# ------------------------------------------------------------------

def check_owner_wrapper(handler):
    def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id == BROADCAST_ADMIN_ID:
            return handler(update, context)
        else:
            update.message.reply_text(r"❌ You are not authorized to use this command\.", parse_mode='MarkdownV2')
    return wrapper
    
@check_owner_wrapper
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(r"Usage: `/broadcast Your message here`", parse_mode='MarkdownV2')
        return

    message_to_send = " ".join(context.args)
    sent_count = 0
    fail_count = 0
    
    await update.message.reply_text(rf"Starting broadcast to {len(known_users)} chats\.\.\.", parse_mode='MarkdownV2')
    
    escaped_broadcast_message = escape_markdown_v2(message_to_send)
    
    for chat_id in list(known_users):
        try:
            final_broadcast_message = rf"**📣 B R O A D C A S T**\n\n{escaped_broadcast_message}"
            await context.bot.send_message(chat_id, final_broadcast_message, parse_mode='MarkdownV2') 
            sent_count += 1
        except Exception as e:
            logger.warning(f"Failed to send message to chat {chat_id}: {e}")
            fail_count += 1
            await asyncio.sleep(0.1)

    await update.message.reply_text(
        r"**✅ Broadcast Finished\.**\n"
        rf"**Sent**: {sent_count}\n"
        rf"**Failed**: {fail_count}",
        parse_mode='MarkdownV2' 
    )


async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            save_chat_id(update.effective_chat.id)
            await context.bot.send_message(
                update.effective_chat.id,
                r"Thank you for adding me\! Use `/start` to see a welcome message or `/game` to start the challenge\. I need `Send Messages` permission to work correctly\.",
                parse_mode='MarkdownV2' 
            )
            return

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global total_messages_processed
    total_messages_processed += 1
    
    text = update.effective_message.text or update.effective_message.caption
    
    if not text or text.startswith('/'): 
        if text and text.lower().startswith('/photoid'):
            await photoid_command(update, context)
        return
        
    await check_guess_and_update_score(update, context, text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id in active_games or update.effective_chat.id in rapid_games:
        await update.message.reply_text(r"Sorry, I currently cannot process voice messages for game answers\. Please type your guess\.", parse_mode='MarkdownV2')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("A critical error occurred:", exc_info=context.error)
    if update and update.effective_chat:
        try:
             await context.bot.send_message(
                update.effective_chat.id,
                r"⚠️ **An unexpected error occurred\.** The developer has been notified\. Please try again or use `/help`\.",
                parse_mode='MarkdownV2' 
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
    
    conn, error = db_connect()
    if not conn:
        logger.critical(f"CRITICAL: Failed to connect to PostgreSQL on startup. {error}. Check your DATABASE_URL.")
    
    load_known_users()
    
    if not PEXELS_API_KEY and not UNSPLASH_ACCESS_KEY:
        logger.warning("WARNING: Both PEXELS_API_KEY and UNSPLASH_ACCESS_KEY are missing. Image guessing game will not work.")
    
    if not GEMINI_API_KEY:
        logger.warning("WARNING: GEMINI_API_KEY is missing. AI-based answer generation will fail.")

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
    application.add_handler(CommandHandler("rapidquiz", rapidquiz_command)) 
    application.add_handler(CommandHandler("myscore", my_score_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("album", album_command)) 
    application.add_handler(CommandHandler("collection", album_command)) 
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("hint", simple_hint_command))
    application.add_handler(CommandHandler("howtoplay", howtoplay_command)) 
    application.add_handler(CommandHandler("photoid", photoid_command)) 
    
    # Owner-Only/Utility Commands
    application.add_handler(CommandHandler("broadcast", broadcast_command)) 
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
        logger.info("Bot started with polling")
        application.run_polling(poll_interval=1)

if __name__ == '__main__':
    main()
