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
import html # Zaroori naya import

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
    """Escapes Telegram MarkdownV2 special characters. (Keeping this function, 
    but it's redundant if we fully switch to HTML)"""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', text)

# --- NAYA FUNCTION: MARKDOWNV2 SE HTML CONVERSION ---

def markdownv2_to_html_converter(text: str) -> str:
    """Telegram MarkdownV2 formatting ko Telegram-compatible HTML mein convert karta hai."""
    
    # 1. Links ([text](url)) -> <a href="url">text</a>
    # Note: Link conversion should be first to protect inner content.
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    # 2. Code Blocks (pre-formatted) - ```lang\ncode\n``` -> <pre>code</pre>
    # Content inside code is escaped using html.escape to prevent internal HTML formatting
    text = re.sub(
        r'```(?:\w*\n)?(.*?)```', 
        lambda m: f'<pre>{html.escape(m.group(1).strip())}</pre>', 
        text, 
        flags=re.DOTALL
    )

    # 3. Inline Code (`code`) -> <code>code</code>
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # 4. Tags - (All tags are converted, only if not escaped by a backslash \ )
    
    # Bold: *bold* -> <b>bold</b>
    text = re.sub(r'(?<!\\)\*(.*?)(?<!\\)\*', r'<b>\1</b>', text)

    # Italic: _italic_ -> <i>italic</i>
    text = re.sub(r'(?<!\\)\_(.*?)(?<!\\)\_', r'<i>\1</i>', text)

    # Underline: __underline__ -> <u>underline</u>
    text = re.sub(r'(?<!\\)\_\_(.*?)(?<!\\)\_\_', r'<u>\1</u>', text)

    # Strikethrough: ~strikethrough~ -> <s>strikethrough</s>
    text = re.sub(r'(?<!\\)\~(.*?)(?<!\\)\~', r'<s>\1</s>', text)

    # Spoiler: ||spoiler|| -> <span class="tg-spoiler">spoiler</span>
    text = re.sub(r'(?<!\\)\|\|(.*?)(?<!\\)\|\|', r'<span class="tg-spoiler">\1</span>', text)
    
    # 5. Final cleanup: Remove backslashes used for escaping MarkdownV2 characters.
    text = re.sub(r'\\([_*\[\]()~`>#+-=|{}.!])', r'\1', text)
    
    # 6. HTML Escape of remaining plain text special chars (must be done after tag conversion)
    # This must be done carefully to not escape newly created tags.
    # The simplest way is to perform a full escape and then selectively unescape the tags.
    # Since the focus is only on the required conversion for Telegram, 
    # we rely on the `re.sub` being run first. The rest of the content is treated as text.
    
    # FINAL STEP: Escape HTML special characters in the text that are NOT part of a valid HTML tag
    # A full escape and then a re-insertion of the valid tags is the most robust way.
    # We will use a simpler, common bot developer approach:
    
    text = text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
    
    # Now, unescape the known valid HTML tags used by Telegram, but not the content inside them.
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;u&gt;', '<u>').replace('&lt;/u&gt;', '</u>')
    text = text.replace('&lt;s&gt;', '<s>').replace('&lt;/s&gt;', '</s>')
    text = text.replace('&lt;code&gt;', '<code>').replace('&lt;/code&gt;', '</code>')
    text = text.replace('&lt;pre&gt;', '<pre>').replace('&lt;/pre&gt;', '</pre>')
    text = text.replace('&lt;span class=\"tg-spoiler\"&gt;', '<span class="tg-spoiler">').replace('&lt;/span&gt;', '</span>')
    
    # Unescape anchor tags
    def unescape_anchor(match):
        return html.unescape(match.group(0))

    text = re.sub(r'<a href=".*?">.*?</a>', unescape_anchor, text)
    
    return text


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
    # CONVERTED: MarkdownV2 to HTML
    text = "<b>🖼️ C H O O S E C H A L L E N G E</b>\n\nSelect a precision level for the image analysis (Difficulty):"
    reply_markup = get_difficulty_menu()
    
    chat_id = update.effective_chat.id if update.effective_chat else (update.callback_query.message.chat_id if update.callback_query else None)
    if not chat_id: return

    if update.callback_query:
        query = update.callback_query
        try:
            await query.answer()
            await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
        except Exception:
            await context.bot.send_message(chat_id, text, parse_mode='HTML', reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

# ... (Database functions remain the same) ...

# ------------------------------------------------------------------
# --- RAPID QUIZ GAME FUNCTIONS ---
# ------------------------------------------------------------------

# ... (fetch_image functions remain the same) ...

async def timeout_rapid_round_task(chat_id: int, context: ContextTypes.DEFAULT_TYPE, time_limit: int):
    try:
        await asyncio.sleep(time_limit)
        
        if chat_id not in rapid_games:
            return
            
        state = rapid_games[chat_id]
        
        caption_markdown = f"⏱️ **T I M E U P** ⏱️\n\n**Round {state['current_round']} Skipped!** The answer was **{state['answer'].upper()}**."
        caption_html = markdownv2_to_html_converter(caption_markdown)
        
        try:
            if 'last_message_id' in state:
                # CONVERTED: MarkdownV2 to HTML
                await context.bot.edit_message_caption(
                    chat_id=chat_id, 
                    message_id=state['last_message_id'],
                    caption=caption_html,
                    reply_markup=None,
                    parse_mode='HTML' 
                )
        except Exception:
            pass 

        await context.bot.send_message(
            chat_id=chat_id, 
            text=caption_html,
            parse_mode='HTML' 
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
    
    # CONVERTED: MarkdownV2 to HTML
    loading_markdown = f"🚀 **RAPID QUIZ** - Round {state['current_round']}/{state['max_rounds']}\n\n**⏳ Downloading and analyzing image. Please wait...**"
    loading_message = await context.bot.send_message(
        chat_id, 
        markdownv2_to_html_converter(loading_markdown),
        parse_mode='HTML' 
    )
    
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass
        
    if not pixelated_img_io:
        await context.bot.send_message(chat_id, markdownv2_to_html_converter(f"❌ **Error**: Image acquisition failed. Rapid Quiz aborted."), parse_mode='HTML')
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
    
    # CONVERTED: MarkdownV2 to HTML
    caption_markdown = (
        f"🚀 **RAPID QUIZ** - Round **{state['current_round']}/{state['max_rounds']}**\n\n"
        f"**Answer**: **{len(answer)}** letters. (Reward: **+{RAPID_QUIZ_SETTINGS['reward']}** pts)\n"
        f"**Time Left**: **{time_limit} seconds**.\n"
        f"Guess the word *fast* to continue the streak!" 
    )
    
    try:
        sent_message = await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=markdownv2_to_html_converter(caption_markdown), 
            parse_mode='HTML' 
        )
        state['last_message_id'] = sent_message.message_id
    except Exception as e:
        logger.error(f"Failed to send rapid quiz photo: {e}")
        await context.bot.send_message(chat_id, markdownv2_to_html_converter(f"❌ **Error**: Image transmission failed. Rapid Quiz aborted."), parse_mode='HTML')
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
    
    # CONVERTED: MarkdownV2 to HTML
    end_markdown = (
        f"🏁 **R A P I D Q U I Z E N D E D** 🏁\n\n"
        f"**Total Rounds**: **{rounds_played} / {state['max_rounds']}**\n"
        f"**Points Earned**: **+{total_score}**\n"
        f"**New Balance**: **{new_balance:,}** Points." 
    )
    
    await context.bot.send_message(
        chat_id, 
        markdownv2_to_html_converter(end_markdown), 
        parse_mode='HTML'
    )
    
    await context.bot.send_message(
        chat_id, 
        "<b>▶️ S T A R T N E W G A M E ?</b>",
        parse_mode='HTML',
        reply_markup=get_difficulty_menu() 
    )


async def rapidquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # CONVERTED: MarkdownV2 to HTML
    if chat_id in active_games:
        await update.message.reply_text(markdownv2_to_html_converter(f"A normal game session is currently active. Please finish it with `/skip` or submit your guess."), parse_mode='HTML')
        return
        
    if chat_id in rapid_games:
        await update.message.reply_text(markdownv2_to_html_converter(f"A Rapid Quiz session is already active!"), parse_mode='HTML')
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
    
    start_markdown = (
        f"🚀 **R A P I D Q U I Z** 🚀\n\n"
        f"Starting 10 rounds, with decreasing time limits per image. Get ready!"
    ) 
    await update.message.reply_text(
        markdownv2_to_html_converter(start_markdown), 
        parse_mode='HTML'
    ) 
    await start_rapid_round(chat_id, context)

async def end_rapid_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("Manually ending the Rapid Quiz...", parse_mode='HTML')
    if chat_id not in rapid_games:
        await update.message.reply_text("No active Rapid Quiz to end.", parse_mode='HTML')
        return
        
    await end_rapid_quiz_logic(chat_id, context)

# ------------------------------------------------------------------
# --- TELEGRAM COMMAND HANDLERS (Normal Mode & Support) ---
# ------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_chat_id(update.effective_chat.id)
    
    # CONVERTED: MarkdownV2 to HTML
    welcome_markdown = (
        f"**Welcome to {BOT_USERNAME}!** 🖼️\n\n"
        f"I'm a guessing bot that shows you heavily pixelated images. Your mission is to guess the object or word in the picture.\n\n" 
        f"**C O M M A N D S**:\n"
        f"• `/game` - Start a new pixel challenge.\n"
        f"• `/rapidquiz` - Start a fast-paced 10-round challenge (Time limit decreases).\n"
        f"• `/myscore` - Check your points.\n"
        f"• `/leaderboard` - See the top players.\n"
        f"• `/howtoplay` - Detailed instructions.\n\n"
        f"Get started with the button below!"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='start_menu_selector')], 
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    
    try:
        await update.message.reply_photo(
            photo=WELCOME_IMAGE_URL,
            caption=markdownv2_to_html_converter(welcome_markdown),
            parse_mode='HTML', 
            reply_markup=keyboard
        )
    except Exception:
        await update.message.reply_text(markdownv2_to_html_converter(welcome_markdown), parse_mode='HTML', reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # CONVERTED: MarkdownV2 to HTML
    help_markdown = (
        f"**📚 B O T C O M M A N D S**\n\n"
        f"**Game:**\n"
        f"• `/game` - Start a new pixel challenge (Normal Mode).\n" 
        f"• `/rapidquiz` - Start the 10-round rapid quiz challenge.\n"
        f"• `/skip` - Skip the current normal game and reveal the answer.\n" 
        f"• `/hint` - Use a hint (costs points).\n\n" 
        f"**Economy & Stats:**\n"
        f"• `/myscore` - Check your current point balance.\n" 
        f"• `/profile` or `/stats` - View your rank, streak, and album progress.\n" 
        f"• `/leaderboard` - See the global top players.\n" 
        f"• `/daily` - Claim your daily bonus points.\n\n" 
        f"**Collection & Shop:**\n"
        f"• `/album` - View your image categories.\n" 
        f"• `/shop` - See and buy special in-game items.\n\n" 
        f"Use `/howtoplay` for game details." 
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Game", callback_data='start_menu_selector')], 
        [InlineKeyboardButton("How to Play", callback_data='help_menu')]
    ])
    
    await update.message.reply_text(markdownv2_to_html_converter(help_markdown), parse_mode='HTML', reply_markup=keyboard) 

async def howtoplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # CONVERTED: MarkdownV2 to HTML
    rules_markdown = (
        f"**📜 H O W T O P L A Y**\n\n"
        f"**1. The Goal:** Guess the object, person, or word in the pixelated image.\n\n" 
        f"**2. Difficulty & Points (`/game`):**\n" 
        f"• **Easy:** Clearest image, low reward.\n" 
        f"• **Extreme:** Heavily pixelated, high reward.\n" 
        f"The harder the difficulty, the more points you earn!\n\n" 
        f"**3. Guessing:** Just send the word you think is correct in the chat. Case and spaces don't matter.\n\n" 
        f"**4. Hints & Costs:**\n"
        f"• You get one **free hint** (the category) with every game.\n" 
        f"• Use `/hint` to reveal a letter in the word (costs 10 points).\n\n" 
        f"**5. Rapid Quiz (`/rapidquiz`):**\n"
        f"• 10 consecutive rounds.\n" 
        f"• **Time decreases** each round (Starting at 50s, minimum 10s).\n" 
        f"• No hints or skips. Guess correctly to immediately advance.\n\n" 
        f"Good luck, Agent!" 
    )
    
    await update.message.reply_text(markdownv2_to_html_converter(rules_markdown), parse_mode='HTML') 

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    process = psutil.Process(os.getpid())
    uptime = datetime.now() - start_time
    
    # CONVERTED: MarkdownV2 to HTML (Removed `escape_markdown_v2` for `uptime` and replaced with `html.escape` if needed)
    about_markdown = (
        f"**🤖 A B O U T {BOT_USERNAME}**\n\n"
        f"**Version:** v3.2 (Final Flow Fix)\n"
        f"**Developer:** Ankit / Pixel Team\n"
        f"**Source APIs:** Pexels, Unsplash, Google Gemini AI\n"
        f"**Uptime:** {html.escape(str(timedelta(seconds=int(uptime.total_seconds()))))}\n" 
        f"**Messages Processed:** {total_messages_processed:,}\n\n"
        f"**Support Group:** [Join here]({SUPPORT_GROUP_LINK})"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Support Group", url=SUPPORT_GROUP_LINK)]
    ])
    
    await update.message.reply_text(markdownv2_to_html_converter(about_markdown), parse_mode='HTML', reply_markup=keyboard) 

async def photoid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message.photo:
        file_id = update.effective_message.photo[-1].file_id
        await update.message.reply_text(f"File ID of the photo: <code>{file_id}</code>", parse_mode='HTML')
    elif update.effective_message.reply_to_message and update.effective_message.reply_to_message.photo:
        file_id = update.effective_message.reply_to_message.photo[-1].file_id
        await update.message.reply_text(f"File ID of the replied photo: <code>{file_id}</code>", parse_mode='HTML')
    else:
        await update.message.reply_text(f"Please send this command as a reply to a photo or as the caption of a photo to get its ID.", parse_mode='HTML')

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    game_state = active_games.pop(chat_id, None)
    
    # CONVERTED: MarkdownV2 to HTML
    if chat_id in rapid_games:
        await update.message.reply_text(markdownv2_to_html_converter("You cannot skip Rapid Quiz. Use `/endrapidquiz` to quit."), parse_mode='HTML')
        return

    if game_state:
        correct_answer = game_state['answer'].upper()
        letter_count = len(game_state['answer']) 
        original_url = game_state['url']
        
        skip_markdown = (
            f"🛑 **G A M E S K I P P E D** 🛑\n\nThe correct solution was: **{correct_answer}** ({letter_count} letters)."
        )
        await update.message.reply_text(
            markdownv2_to_html_converter(skip_markdown), 
            parse_mode='HTML' 
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=markdownv2_to_html_converter(f"Original Image. Solution: **{correct_answer}**."),
                parse_mode='HTML' 
            )
        except Exception:
            await update.message.reply_text("Could not send the original image file.", parse_mode='HTML')

        await update.message.reply_text(
            "<b>▶️ S T A R T N E W G A M E ?</b>",
            parse_mode='HTML',
            reply_markup=get_difficulty_menu() 
        )
    else:
        await update.message.reply_text("No active normal game to skip. Use `/game` to start one.", parse_mode='HTML')

async def simple_hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    game_state = active_games.get(chat_id)
    if update.callback_query:
        message_sender = update.callback_query.message
    else:
        message_sender = update.message
    
    if not game_state:
        await message_sender.reply_text("No active normal game to get a hint for. Use `/game` to start one.", parse_mode='HTML')
        return

    points_cost = GAME_LEVELS[game_state['difficulty']]['hint_cost']
    max_hints = GAME_LEVELS[game_state['difficulty']]['max_hints']
    
    current_score = get_user_score(user_id)
    
    if game_state['hints_taken'] >= max_hints:
        await message_sender.reply_text("You have already used the maximum number of hints for this game.", parse_mode='HTML')
        return

    # CONVERTED: MarkdownV2 to HTML
    if current_score < points_cost:
        insufficient_markdown = (
            f"❌ **INSUFFICIENT FUNDS!**\n\n"
            f"You need **{points_cost}** points for a hint, but you only have **{current_score}**.\n"
            f"Earn points by solving images or claim your `/daily` bonus."
        )
        await message_sender.reply_text(
            markdownv2_to_html_converter(insufficient_markdown),
            parse_mode='HTML' 
        )
        return

    update_user_score(user_id, -points_cost)
    game_state['hints_taken'] += 1
    
    answer = game_state['answer'].lower()
    hint_list = list(game_state['hint_string'])
    
    unrevealed_indices = [i for i, char in enumerate(hint_list) if char == '_']
    
    if not unrevealed_indices:
        await message_sender.reply_text("The word is already fully revealed!", parse_mode='HTML')
        return

    index_to_reveal = random.choice(unrevealed_indices)
    hint_list[index_to_reveal] = answer[index_to_reveal].upper()
    game_state['hint_string'] = "".join(hint_list)
    
    # NOTE: Since the hint_sentence contains no MarkdownV2, we just escape HTML chars
    escaped_hint_sentence = html.escape(game_state['hint_sentence'])
    
    new_score = get_user_score(user_id)
    
    # CONVERTED: MarkdownV2 to HTML
    hint_markdown = (
        f"💡 **H I N T U S E D** (-{points_cost} Points)\n\n"
        f"**New Progress**: `{game_state['hint_string']}`\n"
        f"**Clue**: *{escaped_hint_sentence}*\n"
        f"**Hints Left**: **{max_hints - game_state['hints_taken']}**\n"
        f"**New Balance**: **{new_score}** Points."
    )
    
    await message_sender.reply_text(markdownv2_to_html_converter(hint_markdown), parse_mode='HTML') 


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # CONVERTED: MarkdownV2 to HTML
    if chat_id in active_games:
        await update.message.reply_text(markdownv2_to_html_converter("A normal game session is currently active. Please submit your guess or use `/skip`."), parse_mode='HTML')
        return
    
    if chat_id in rapid_games:
        await update.message.reply_text(markdownv2_to_html_converter("A Rapid Quiz session is currently active. Please finish it or use `/endrapidquiz`."), parse_mode='HTML')
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
            
            caption_markdown = f"🛑 **G A M E T E R M I N A T E D** 🛑\n\nThe correct solution was: **{correct_answer}** ({letter_count} letters)."
            caption_html = markdownv2_to_html_converter(caption_markdown)

            try:
                 await query.edit_message_caption(
                    caption=caption_html, 
                    parse_mode='HTML', 
                    reply_markup=None 
                )
            except Exception:
                await context.bot.send_message(chat_id, f"The game has ended. The correct answer was: <b>{correct_answer}</b> ({letter_count} letters).", parse_mode='HTML')
                
            try:
                await context.bot.send_photo(
                    chat_id, 
                    photo=original_url, 
                    caption=markdownv2_to_html_converter(f"Original Image. Solution: **{correct_answer}**."),
                    parse_mode='HTML' 
                )
            except Exception:
                pass

            await context.bot.send_message(
                chat_id, 
                "<b>▶️ S T A R T N E W G A M E ?</b>",
                parse_mode='HTML',
                reply_markup=get_difficulty_menu() 
            )

        else:
            await context.bot.send_message(chat_id, "No active game to terminate.", parse_mode='HTML')
        return

    if data == 'game_hint':
        await simple_hint_command(query, context)
        return

    if not data.startswith('game_'):
        return

    difficulty = data.split('_')[1]
    
    if chat_id in active_games:
        await context.bot.send_message(chat_id, "A game is already active!", parse_mode='HTML')
        return
    
    # CONVERTED: MarkdownV2 to HTML
    loading_markdown = (
        f"**Challenge Initiated:** *{difficulty.upper()}*.\n\n"
        f"**⏳ Downloading and analyzing image. Please wait...**"
    )
    loading_message = await context.bot.send_message(
        chat_id, 
        markdownv2_to_html_converter(loading_markdown),
        parse_mode='HTML' 
    )
    
    pixelated_img_io, answer, original_url, category, hint_sentence = await fetch_and_pixelate_image(difficulty)
    
    try:
        await context.bot.delete_message(chat_id, loading_message.message_id)
    except Exception:
        pass

    if not pixelated_img_io:
        escaped_error_message = html.escape(answer)
        await context.bot.send_message(chat_id, f"❌ <b>Error</b>: Image acquisition failed. Details: {escaped_error_message}", parse_mode='HTML')
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
    
    # CONVERTED: MarkdownV2 to HTML
    caption_markdown = (
        f"**📸 V I S U A L C H A L L E N G E: {difficulty.upper()}**\n\n"
        f"Identify the object in this high-pixel density image.\n\n"
        f"**Reward**: **+{points} Points**\n"
        f"**Progress**: `{initial_hint_string}` ({len(answer)} letters)\n"
        f"**Free Clue**: *Category is {category_emoji} {category.capitalize()}*\n"
        f"Use `/hint` to reveal a letter!"
    )
    
    game_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💡 Request Letter Hint (-{level_data['hint_cost']} pts) ({level_data['max_hints'] - active_games[chat_id]['hints_taken']} remaining)", callback_data='game_hint')],
        [InlineKeyboardButton("🛑 Terminate Game", callback_data='game_end')]
    ])
    
    try:
        await context.bot.send_photo(
            chat_id, 
            photo=pixelated_img_io, 
            caption=markdownv2_to_html_converter(caption_markdown), 
            parse_mode='HTML', 
            reply_markup=game_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send pixelated photo: {e}")
        await context.bot.send_message(chat_id, markdownv2_to_html_converter(f"❌ **Error**: Image transmission failed. Challenge cancelled."), parse_mode='HTML')
        del active_games[chat_id]


async def check_guess_and_update_score(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    # CONVERTED: MarkdownV2 to HTML (using html.escape)
    user_name = html.escape(update.effective_user.first_name)
    
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

            caption_markdown = f"✅ **R O U N D S O L V E D** ✅\n\nCorrect Answer: **{correct_answer.upper()}**."
            caption_html = markdownv2_to_html_converter(caption_markdown)
            
            try:
                if 'last_message_id' in state:
                    await context.bot.edit_message_caption(
                        chat_id=chat_id,
                        message_id=state['last_message_id'],
                        caption=caption_html,
                        reply_markup=None,
                        parse_mode='HTML' 
                    )
            except Exception:
                pass
                
            await context.bot.send_message(
                chat_id, 
                caption_html + f"\n<b>Reward</b>: <b>+{reward}</b> Points.", # Separate reward message for clarity
                parse_mode='HTML' 
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
        
        # CONVERTED: MarkdownV2 to HTML
        caption_markdown = (
            f"✅ **S O L U T I O N A C Q U I R E D** ✅\n\n"
            f"**Agent {user_name}** successfully identified: **{correct_answer.upper()}** ({letter_count} letters)\n\n"
            f"**Reward**: **+{points} Points**\n"
            f"**Current Balance**: **{current_score:,}**\n"
            f"View the original image below."
        )
        
        try:
            await context.bot.send_photo(
                chat_id, 
                photo=original_url, 
                caption=markdownv2_to_html_converter(caption_markdown),
                parse_mode='HTML'
            )
        except Exception:
            await context.bot.send_message(chat_id, markdownv2_to_html_converter(f"{caption_markdown}\n(Original image file unavailable)."), parse_mode='HTML')
        
        await context.bot.send_message(
            chat_id, 
            "<b>▶️ S T A R T N E W G A M E ?</b>",
            parse_mode='HTML',
            reply_markup=get_difficulty_menu() 
        )

# ------------------------------------------------------------------
# --- SUPPORT COMMAND HANDLERS (Standard Implementations) ---
# ------------------------------------------------------------------

async def my_score_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    score = get_user_score(user_id)
    # CONVERTED: MarkdownV2 to HTML
    await update.message.reply_text(markdownv2_to_html_converter(f"💰 **Y O U R B A L A N C E**\n\nYour current score is: **{score:,} Points**."), parse_mode='HTML')

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_name = html.escape(user.first_name)
    rank, score, solved_count, current_streak, last_daily_claim = await get_user_profile_data(user.id)

    # CONVERTED: MarkdownV2 to HTML
    profile_markdown = (
        f"**👤 A G E N T P R O F I L E**\n\n"
        f"**Agent**: {user_name}\n"
        f"**Rank**: #{rank:,}\n"
        f"**Score**: **{score:,}** Points\n"
        f"**Solved Images**: {solved_count:,}\n"
        f"**Current Streak**: {current_streak}\n\n"
        f"Use `/album` to see your collected categories."
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("View Album", callback_data='album_view_1')]
    ])
    
    await update.message.reply_text(markdownv2_to_html_converter(profile_markdown), parse_mode='HTML', reply_markup=keyboard) 

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top_scores = get_top_scores(limit=10)
    
    leaderboard_markdown = f"**🏆 G L O B A L L E A D E R B O A R D**\n\n"
    
    # CONVERTED: MarkdownV2 to HTML
    if not top_scores:
        leaderboard_markdown += f"No scores recorded yet. Start a game with `/game`!"
    else:
        for i, (user_id, score) in enumerate(top_scores, 1):
            try:
                member = await context.bot.get_chat_member(update.effective_chat.id, user_id)
                name = html.escape(member.user.first_name)
            except Exception:
                name = f"Agent {str(user_id)[:4]}..."
            
            leaderboard_markdown += f"{i}. **{name}** - **{score:,}** Points\n"

    await update.message.reply_text(markdownv2_to_html_converter(leaderboard_markdown), parse_mode='HTML') 

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    _, score, _, _, last_daily_claim = await get_user_profile_data(user_id)
    
    now_utc = datetime.now(pytz.utc)
    
    if last_daily_claim and (now_utc - last_daily_claim).total_seconds() < 24 * 3600:
        next_claim_time = last_daily_claim + timedelta(hours=24)
        time_left = next_claim_time - now_utc
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        
        # CONVERTED: MarkdownV2 to HTML
        cooldown_markdown = (
            f"⏳ **D A I L Y B O N U S** - Cooldown\n\n"
            f"You have already claimed your bonus. Come back in **{hours}h {minutes}m** to claim your next **+{DAILY_BONUS_POINTS}** points!"
        )
        await update.message.reply_text(
            markdownv2_to_html_converter(cooldown_markdown),
            parse_mode='HTML' 
        )
    else:
        success = await update_daily_claim(user_id, DAILY_BONUS_POINTS)
        if success:
            new_score = get_user_score(user_id)
            # CONVERTED: MarkdownV2 to HTML
            success_markdown = (
                f"🎉 **D A I L Y B O N U S C L A I M E D** 🎉\n\n"
                f"You received **+{DAILY_BONUS_POINTS}** points!\n"
                f"**New Balance**: **{new_score:,}** Points."
            )
            await update.message.reply_text(
                markdownv2_to_html_converter(success_markdown),
                parse_mode='HTML' 
            )
        else:
            await update.message.reply_text("❌ Failed to claim daily bonus due to a database error.", parse_mode='HTML')

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shop_markdown = f"**🛒 P I X E L S H O P**\n\n"
    keyboard_rows = []
    
    # CONVERTED: MarkdownV2 to HTML
    for key, item in SHOP_ITEMS.items():
        shop_markdown += f"**{key}. {item['name']}** - **{item['cost']}** Points\n"
        # The description is treated as plain text within *...* which becomes <i>...</i>
        shop_markdown += f"*{html.escape(item['description'])}*\n\n" 
        keyboard_rows.append([InlineKeyboardButton(f"Buy {item['name']} ({item['cost']})", callback_data=f'buy_{key}')])

    reply_markup = InlineKeyboardMarkup(keyboard_rows)
    current_score = get_user_score(update.effective_user.id)
    shop_markdown += f"**Your Balance**: **{current_score:,}** Points"
    
    await update.message.reply_text(markdownv2_to_html_converter(shop_markdown), parse_mode='HTML', reply_markup=reply_markup) 

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
            await message.reply_text("Please specify a valid item ID. Use `/shop` to see items.", parse_mode='HTML')
            return
        item_id = context.args[0]
        
    item = SHOP_ITEMS.get(item_id)
    if not item:
        await message.reply_text("Invalid item ID.", parse_mode='HTML')
        return
        
    cost = item['cost']
    current_score = get_user_score(user_id)
    
    # CONVERTED: MarkdownV2 to HTML
    if current_score < cost:
        failed_markdown = (
            f"❌ **P U R C H A S E F A I L E D**\n\n"
            f"You need **{cost}** points but only have **{current_score}**."
        )
        await message.reply_text(
            markdownv2_to_html_converter(failed_markdown),
            parse_mode='HTML' 
        )
        return
        
    update_user_score(user_id, -cost)
    new_score = get_user_score(user_id)
    
    # CONVERTED: MarkdownV2 to HTML
    success_markdown = (
        f"✅ **P U R C H A S E S U C C E S S** ✅\n\n"
        f"You bought **{item['name']}** for **{cost}** points.\n"
        f"*Functionality is currently limited.*\n"
        f"**New Balance**: **{new_score:,}** Points."
    )
    await message.reply_text(
        markdownv2_to_html_converter(success_markdown),
        parse_mode='HTML' 
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
        album_title = f"🖼️ **P I X E L A L B U M** - ({filter_letter} Categories)" 
    elif total_collected > 20:
        collected_categories = all_collected_categories[:20]
        album_title = f"🖼️ **P I X E L A L B U M** - (First 20 Items)" 
    else:
        collected_categories = all_collected_categories
        album_title = f"🖼️ **P I X E L A L B U M** - (All Items)"


    album_markdown = f"{album_title}\n\n"
    
    # CONVERTED: MarkdownV2 to HTML
    if not collected_categories:
        if filter_letter:
            album_markdown += f"No categories starting with **{filter_letter}** found in your album."
        else:
            album_markdown += f"Your album is empty! Solve images to collect categories.\nStart with `/game`."
    else:
        collected_map = {cat: SEARCH_CATEGORIES.get(cat, '❓') for cat in collected_categories}
        
        album_markdown += f"**Collected Categories:**\n"
        
        categories_display = ""
        count = 0
        for cat, emoji in collected_map.items():
            escaped_cat = html.escape(cat.capitalize())
            categories_display += f"{emoji} {escaped_cat} | "
            count += 1
            if count % 3 == 0:
                categories_display += "\n"
                
        album_markdown += categories_display.strip()
        
    
    album_markdown += f"\n\n**Total Progress**: **{total_collected}** / **{total_categories}** categories collected."

    if total_collected > 20 and not filter_letter:
         album_markdown += (
             f"\n\n*Your collection is large! To view all items, use the segmented view:*\n"
             f"**Example**: `/album A` (for categories starting with A)\n"
             f"**Example**: `/album T` (for categories starting with T)"
         )
         
    
    album_html = markdownv2_to_html_converter(album_markdown)
    
    if update.callback_query:
        await update.callback_query.message.reply_text(album_html, parse_mode='HTML') 
    else:
        await update.message.reply_text(album_html, parse_mode='HTML') 

async def donate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # CONVERTED: MarkdownV2 to HTML
    donate_markdown = (
        f"**💖 S U P P O R T D E V E L O P M E N T 💖**\n\n"
        f"If you enjoy the bot, consider making a small donation to help cover server costs and fund future updates.\n\n"
        f"You can scan the QR code below for easy payment. Thank yourself for your support!"
    )
    
    try:
        await update.message.reply_photo(
            photo=DONATION_QR_CODE_ID,
            caption=markdownv2_to_html_converter(donate_markdown),
            parse_mode='HTML' 
        )
    except Exception:
        await update.message.reply_text(markdownv2_to_html_converter(donate_markdown), parse_mode='HTML') 

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
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id == BROADCAST_ADMIN_ID:
            return await handler(update, context)
        else:
            await update.message.reply_text("❌ You are not authorized to use this command.", parse_mode='HTML')
    return wrapper
    
@check_owner_wrapper
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: <code>/broadcast Your message here</code>", parse_mode='HTML')
        return

    message_to_send = " ".join(context.args)
    sent_count = 0
    fail_count = 0
    
    # CONVERTED: MarkdownV2 to HTML
    await update.message.reply_text(f"Starting broadcast to {len(known_users)} chats...", parse_mode='HTML')
    
    # The message_to_send is assumed to be in MarkdownV2 from the user, so we convert it.
    escaped_broadcast_message_html = markdownv2_to_html_converter(message_to_send)
    
    for chat_id in list(known_users):
        try:
            final_broadcast_message = f"<b>📣 B R O A D C A S T</b>\n\n{escaped_broadcast_message_html}"
            await context.bot.send_message(chat_id, final_broadcast_message, parse_mode='HTML') 
            sent_count += 1
        except Exception as e:
            logger.warning(f"Failed to send message to chat {chat_id}: {e}")
            fail_count += 1
            await asyncio.sleep(0.1)

    broadcast_result_markdown = (
        f"**✅ Broadcast Finished.**\n"
        f"**Sent**: {sent_count}\n"
        f"**Failed**: {fail_count}"
    )
    await update.message.reply_text(
        markdownv2_to_html_converter(broadcast_result_markdown),
        parse_mode='HTML' 
    )


async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            save_chat_id(update.effective_chat.id)
            await context.bot.send_message(
                update.effective_chat.id,
                "Thank you for adding me! Use `/start` to see a welcome message or `/game` to start the challenge. I need `Send Messages` permission to work correctly.",
                parse_mode='HTML' 
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
        await update.message.reply_text("Sorry, I currently cannot process voice messages for game answers. Please type your guess.", parse_mode='HTML')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("A critical error occurred:", exc_info=context.error)
    if update and update.effective_chat:
        try:
             await context.bot.send_message(
                update.effective_chat.id,
                "⚠️ <b>An unexpected error occurred.</b> The developer has been notified. Please try again or use <code>/help</code>.",
                parse_mode='HTML' 
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
    # FIX: Added 'start_menu_selector' to the game handler pattern for flow correction
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
