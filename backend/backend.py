from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import func

import asyncio
import os
import json
import re

from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import OpenAI

from database import get_db
from backend.models import Character, GameResult


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
DEFAULT_IMAGE_PATH = "/assets/characters/default.svg"


# ============================================================
# APP
# ============================================================

load_dotenv()

app = FastAPI(title="Who Are You? AI Character Game")


# ============================================================
# GEMINI
# ============================================================

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "You get a completely free day with no responsibilities. Walk me through how you would spend it, and what would make the day feel well spent.",

    "You are trying to learn something difficult and still do not understand it after several attempts. What do you do next, and why?",

    "You have to spend an entire day with a group of strangers. What do you do during the first hour, and what do you pay attention to?",

    "Someone gives you an important task but explains it badly. How do you handle the uncertainty and decide what to do?",

    "Tell me about a time when you changed your mind about something important. What caused the change?",

    "A close friend is about to make a decision you think will cause problems, but they have not asked for your opinion. What would you actually do?",

    "You have two possible solutions to a serious problem: one is safe and predictable, the other could be much better but could fail. How do you decide?",

    "You discover strong evidence that one of your deeply held beliefs is wrong. What happens in your thinking after that?",

    "You can solve a mystery completely, but solving it will permanently remove the mystery and curiosity surrounding it. Would you solve it? Why?",

    "Ignore career, money, reputation, and society's expectations. In your own words, what makes a life worth living?",
]


# ============================================================
# REQUEST MODEL
# ============================================================

class GameRequest(BaseModel):
    answers: List[str] = Field(min_length=10, max_length=10)


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(text: str):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


# ============================================================
# TRAITS
# ============================================================

TRAITS = [
    "curiosity",
    "analytical_thinking",
    "creativity",
    "independence",
    "loyalty",
    "empathy",
    "responsibility",
    "risk_tolerance",
    "adaptability",
    "skepticism",
    "openness",
    "social_energy",
    "directness",
    "conflict_tolerance",
    "humor",
    "ambition",
    "meaning_orientation",
    "autonomy",
    "uncertainty_tolerance",
    "emotional_reflection",
    "practicality",
    "idealism",
    "resourcefulness",
    "protectiveness",
    "justice_orientation",
]


# ============================================================
# PERSONALITY ANALYSIS
# ============================================================

SYSTEM_PROMPT = """You are the semantic personality-analysis engine for a fictional character resemblance game.

Read all 10 answers as one person, not as isolated keyword matches.

Understand:
- intent
- motivations
- reasoning patterns
- values
- behavior
- contradictions
- how the person handles uncertainty

This is a game, not a psychological diagnosis.

Do not diagnose or claim scientific personality certainty.

Return ONLY valid JSON with exactly this shape:

{
  "traits": {"trait_name": 0.0},
  "core_motivations": ["..."],
  "reasoning_style": "...",
  "social_style": "...",
  "important_contradictions": ["..."],
  "evidence": ["brief paraphrase of an answer that supports the analysis", "..."],
  "analysis_confidence": 0.0
}

Use every trait listed by the user.

Scores are numbers from 0 to 1.

Do not select a character.
"""


def analyze_answers_sync(answers: List[str]) -> dict:
    """
    Analyze the user's 10 answers and produce a 25-trait
    personality profile using Gemini.
    """

    user_answers = "\n\n".join(
        f"Question {i + 1}: {QUESTIONS[i]}\n"
        f"Answer: {answers[i]}"
        for i in range(len(answers))
    )

    prompt = f"""
Analyze the following person's answers and create a semantic
personality profile.

This is NOT a medical or psychological diagnosis.

Evaluate all 25 traits from 0.0 to 1.0.

Return ONLY valid JSON in exactly this structure:

{{
  "traits": {{
    "curiosity": 0.0,
    "analytical_thinking": 0.0,
    "creativity": 0.0,
    "independence": 0.0,
    "loyalty": 0.0,
    "empathy": 0.0,
    "responsibility": 0.0,
    "risk_tolerance": 0.0,
    "adaptability": 0.0,
    "skepticism": 0.0,
    "openness": 0.0,
    "social_energy": 0.0,
    "directness": 0.0,
    "conflict_tolerance": 0.0,
    "humor": 0.0,
    "ambition": 0.0,
    "meaning_orientation": 0.0,
    "autonomy": 0.0,
    "uncertainty_tolerance": 0.0,
    "emotional_reflection": 0.0,
    "practicality": 0.0,
    "idealism": 0.0,
    "resourcefulness": 0.0,
    "protectiveness": 0.0,
    "justice_orientation": 0.0
  }},
  "core_motivations": [],
  "reasoning_style": "",
  "social_style": "",
  "important_contradictions": [],
  "evidence": [],
  "analysis_confidence": 0.0
}}

PERSON'S ANSWERS:

{user_answers}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a semantic personality analysis engine. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            reasoning_effort="low"
        )

        text = response.choices[0].message.content or ""

        data = clean_json(text)

        # Make sure every trait exists and is between 0 and 1.
        traits = data.get("traits", {})

        for trait in TRAITS:
            try:
                value = float(traits.get(trait, 0.5))
            except (TypeError, ValueError):
                value = 0.5

            traits[trait] = max(0.0, min(1.0, value))

        data["traits"] = traits

        return data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {e}"
        )
# ============================================================
# CHARACTER DISCOVERY
# ============================================================

CHARACTER_PROMPT = """You are the final character-selection engine for a fictional-character resemblance game.

Given a player's semantic trait profile, consider several well-known fictional characters internally,
then return ONLY the single strongest resemblance.

Characters may come from movies, TV, anime, comics, novels, games, or mythology-inspired fiction.

Do not use a prewritten character database.

This is a game-design similarity estimate, NOT a scientific measurement or diagnosis.

Keep all writing concise:
- exactly 3 why_you_match points
- exactly 2 where_you_differ points

Stats are presentation values from 0 to 100 and should reflect the supplied profile.

Return ONLY valid JSON with this exact shape:

{
  "character_name":"...",
  "universe":"...",
  "match_percentage":91,
  "personality_class":"...",
  "short_description":"...",
  "why_you_match":["...","...","..."],
  "where_you_differ":["...","..."],
  "stats": {
    "analytical_thinking":91,
    "curiosity":96,
    "empathy":72,
    "loyalty":88,
    "independence":84,
    "risk_taking":67,
    "social_energy":55,
    "emotional_reflection":82,
    "creativity":76,
    "justice_orientation":70
  }
}
"""


VISIBLE_STATS = [
    "analytical_thinking",
    "curiosity",
    "empathy",
    "loyalty",
    "independence",
    "risk_taking",
    "social_energy",
    "emotional_reflection",
    "creativity",
    "justice_orientation",
]


# ============================================================
# CHARACTER IMAGES
# ============================================================

DEFAULT_CHARACTER_IMAGE = DEFAULT_IMAGE_PATH

CHARACTER_IMAGE_CACHE = {}


def _get_json(url, headers=None, timeout=4):

    request = Request(
        url,
        headers={
            "User-Agent": "WhoAreYouGame/1.0"
        } | (headers or {})
    )

    with urlopen(
        request,
        timeout=timeout
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


def _normalized(value):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower()
    ).strip()


def _contains_requested(value, requested):

    haystack = _normalized(value)
    needle = _normalized(requested)

    return bool(
        needle
        and (
            needle in haystack
            or haystack in needle
        )
    )


def _character_aliases(character_name):

    name = str(character_name).strip()

    aliases = [name]

    simplified = re.sub(
        r"^(mr|mrs|ms|dr|prof)\.?\s+",
        "",
        name,
        flags=re.IGNORECASE
    )

    if simplified and simplified.lower() != name.lower():
        aliases.append(simplified)

    return aliases


def _looks_like_anime_universe(universe):

    value = _normalized(universe)

    return any(
        keyword in value
        for keyword in (
            "anime",
            "manga",
            "attack on titan",
            "death note",
            "naruto",
            "one piece",
            "dragon ball",
            "demon slayer",
            "my hero academia",
            "jujutsu kaisen",
        )
    )


def _anilist_character_image(character_name, universe):

    if not _looks_like_anime_universe(universe):
        return None

    query = """
    query ($search: String!) {
      Character(search: $search) {
        name { full }
        image { large }
        media(page: 1, perPage: 10) {
          nodes {
            title {
              romaji
              english
              native
            }
          }
        }
      }
    }
    """

    payload = json.dumps(
        {
            "query": query,
            "variables": {
                "search":
                    _character_aliases(character_name)[-1]
            }
        }
    ).encode("utf-8")

    request = Request(
        "https://graphql.anilist.co",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "WhoAreYouGame/1.0",
        }
    )

    with urlopen(
        request,
        timeout=5
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    character = (
        (data.get("data") or {})
        .get("Character")
        or {}
    )

    media = character.get("media") or {}

    titles = []

    for item in media.get("nodes") or []:

        title = item.get("title") or {}

        titles.extend(
            value
            for value in title.values()
            if value
        )

    if (
        not any(
            _contains_requested(
                character.get(
                    "name",
                    {}
                ).get("full", ""),
                alias
            )
            for alias in _character_aliases(character_name)
        )
        or
        not any(
            _contains_requested(
                title,
                universe
            )
            for title in titles
        )
    ):
        return None

    image = (
        character.get("image") or {}
    ).get("large")

    return (
        image
        if isinstance(image, str)
        and image.startswith("https://")
        else None
    )


def _jikan_character_image(character_name, universe):

    if not _looks_like_anime_universe(universe):
        return None

    data = _get_json(
        "https://api.jikan.moe/v4/characters?"
        + urlencode({
            "q":
                _character_aliases(character_name)[-1],
            "limit": 5,
        })
    )

    for result in data.get("data") or []:

        name = result.get("name", "")

        if not any(
            _contains_requested(name, alias)
            for alias in _character_aliases(character_name)
        ):
            continue

        full = _get_json(
            f"https://api.jikan.moe/v4/characters/"
            f"{result.get('mal_id')}/full"
        )

        anime_titles = [
            item.get("title", "")
            for item in
            full.get("data", {}).get("anime") or []
        ]

        if not any(
            _contains_requested(title, universe)
            for title in anime_titles
        ):
            continue

        image = (
            (result.get("images") or {})
            .get("jpg") or {}
        ).get("image_url")

        if (
            isinstance(image, str)
            and image.startswith("https://")
        ):
            return image

    return None


def _tmdb_character_image(character_name, universe):

    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        return None

    data = _get_json(
        "https://api.themoviedb.org/3/search/multi?"
        + urlencode({
            "api_key": api_key,
            "query":
                f"{character_name} {universe}",
            "include_adult": "false",
        })
    )

    for result in data.get("results") or []:

        media_type = result.get("media_type")

        if media_type not in ("movie", "tv"):
            continue

        title = (
            result.get("title")
            or result.get("name")
            or ""
        )

        if not _contains_requested(
            title,
            universe
        ):
            continue

        credits = _get_json(
            f"https://api.themoviedb.org/3/"
            f"{media_type}/"
            f"{result.get('id')}/credits?"
            + urlencode({
                "api_key": api_key
            })
        )

        for cast in credits.get("cast") or []:

            role_names = cast.get(
                "character"
            ) or ""

            if (
                _contains_requested(
                    role_names,
                    character_name
                )
                and cast.get("profile_path")
            ):

                return (
                    "https://image.tmdb.org/t/p/w780"
                    + cast["profile_path"]
                )

    return None


def _wikipedia_character_image(
    character_name,
    universe
):

    search_names = _character_aliases(
        character_name
    )

    data = _get_json(
        "https://en.wikipedia.org/w/api.php?"
        + urlencode({
            "action": "query",
            "list": "search",
            "srsearch":
                f"{search_names[-1]} {universe}",
            "srnamespace": 0,
            "srlimit": 5,
            "format": "json",
        })
    )

    for result in (
        (data.get("query") or {})
        .get("search")
        or []
    ):

        title = result.get(
            "title",
            ""
        )

        searchable_text = (
            title
            + " "
            + result.get(
                "snippet",
                ""
            )
        )

        exact_character_page = any(
            _normalized(title)
            == _normalized(alias)
            for alias in search_names
        )

        universe_match = (
            _contains_requested(
                searchable_text,
                universe
            )
            or exact_character_page
        )

        if not (
            any(
                _contains_requested(
                    title,
                    alias
                )
                for alias in search_names
            )
            and universe_match
        ):
            continue

        page = _get_json(
            "https://en.wikipedia.org/w/api.php?"
            + urlencode({
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "piprop":
                    "original|thumbnail",
                "pithumbsize": 1000,
                "format": "json",
            })
        )

        pages = (
            (page.get("query") or {})
            .get("pages")
            or {}
        ).values()

        for item in pages:

            image = (
                (item.get("original") or {})
                .get("source")
                or
                (item.get("thumbnail") or {})
                .get("source")
            )

            if (
                isinstance(image, str)
                and image.startswith("https://")
            ):
                return image

    return None


def get_character_image(character_name: str, universe: str) -> str:
    """
    Find a character image using fast external sources.

    Order:
    1. AniList for anime/manga characters
    2. Wikipedia
    3. Default local image

    The result is cached in memory so repeated requests
    for the same character do not repeat external searches.
    """

    cache_key = f"{_normalized(character_name)}|{_normalized(universe)}"

    # Return cached image immediately.
    if cache_key in CHARACTER_IMAGE_CACHE:
        return CHARACTER_IMAGE_CACHE[cache_key]

    # --------------------------------------------------------
    # 1. AniList
    # --------------------------------------------------------

    try:
        if _looks_like_anime_universe(universe):
            image = _anilist_character_image(
                character_name,
                universe
            )

            if image:
                CHARACTER_IMAGE_CACHE[cache_key] = image
                return image

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Wikipedia
    # --------------------------------------------------------

    try:
        image = _wikipedia_character_image(
            character_name,
            universe
        )

        if image:
            CHARACTER_IMAGE_CACHE[cache_key] = image
            return image

    except Exception:
        pass

    # --------------------------------------------------------
    # 3. Default image
    # --------------------------------------------------------

    return DEFAULT_CHARACTER_IMAGE


# ============================================================
# NORMALIZE CHARACTER
# ============================================================

def normalize_character(
    data,
    profile
):

    candidate = data

    if (
        isinstance(data, dict)
        and data.get("candidates")
    ):
        candidate = data["candidates"][0]

    candidate = (
        candidate
        if isinstance(candidate, dict)
        else {}
    )

    source_traits = profile.get(
        "traits",
        {}
    )

    stats = (
        candidate.get("stats", {})
        if isinstance(
            candidate.get("stats"),
            dict
        )
        else {}
    )

    trait_aliases = {
        "risk_taking":
            "risk_tolerance"
    }

    for stat in VISIBLE_STATS:

        source_name = (
            trait_aliases.get(
                stat,
                stat
            )
        )

        value = stats.get(
            stat,
            source_traits.get(
                source_name,
                0.5
            ) * 100
        )

        try:

            stats[stat] = max(
                0,
                min(
                    100,
                    round(float(value))
                )
            )

        except (
            TypeError,
            ValueError
        ):

            stats[stat] = 50

    match_value = candidate.get(
        "match_percentage",
        candidate.get(
            "match_score",
            0
        )
    )

    try:

        match_value = max(
            0,
            min(
                100,
                round(float(match_value))
            )
        )

    except (
        TypeError,
        ValueError
    ):

        match_value = 0

    def as_list(value):

        if isinstance(value, list):
            return [
                str(item)
                for item in value
            ]

        if value:
            return [str(value)]

        return []

    return {

        "character_name":
            str(
                candidate.get(
                    "character_name",
                    candidate.get(
                        "name",
                        "Unknown Character"
                    )
                )
            ),

        "universe":
            str(
                candidate.get(
                    "universe",
                    candidate.get(
                        "work",
                        "Unknown universe"
                    )
                )
            ),

        "match_percentage":
            match_value,

        "personality_class":
            str(
                candidate.get(
                    "personality_class",
                    "The Unwritten Archetype"
                )
            ),

        "short_description":
            str(
                candidate.get(
                    "short_description",
                    candidate.get(
                        "why_they_match",
                        "A distinctive fictional resemblance."
                    )
                )
            ),

        "why_you_match":
            as_list(
                candidate.get(
                    "why_you_match"
                )
                or candidate.get(
                    "why_they_match"
                )
            )[:4],

        "where_you_differ":
            as_list(
                candidate.get(
                    "where_you_differ"
                )
                or candidate.get(
                    "why_they_do_not_fully_match"
                )
            )[:3],

        "stats":
            stats,
    }


def discover_characters_sync(profile: dict) -> dict:
    """
    Use Gemini to select the single fictional character
    whose personality most closely matches the analyzed profile.
    """

    character_model = os.getenv(
        "GEMINI_CHARACTER_MODEL",
        MODEL
    )

    # Only send the information needed for character matching.
    compact_profile = {
        "traits": profile.get("traits", {}),
        "core_motivations": profile.get("core_motivations", []),
        "reasoning_style": profile.get("reasoning_style", ""),
        "social_style": profile.get("social_style", ""),
        "important_contradictions": profile.get(
            "important_contradictions", []
        ),
    }

    prompt = f"""
Analyze this personality profile and select ONE fictional character
whose personality most closely resembles it.

Do not choose based on appearance, popularity, fame, or superficial similarities.

Compare:
- personality traits
- motivations
- reasoning style
- social behavior
- emotional tendencies
- contradictions

Return ONLY valid JSON.

Required format:

{{
  "character_name": "...",
  "universe": "...",
  "match_percentage": 0,
  "personality_class": "...",
  "short_description": "...",
  "why_you_match": [
    "...",
    "...",
    "..."
  ],
  "where_you_differ": [
    "...",
    "..."
  ],
  "stats": {{
    "analytical_thinking": 0,
    "curiosity": 0,
    "empathy": 0,
    "loyalty": 0,
    "independence": 0,
    "risk_taking": 0,
    "social_energy": 0,
    "emotional_reflection": 0,
    "creativity": 0,
    "justice_orientation": 0
  }}
}}

All percentage/stat values must be between 0 and 100.

PERSONALITY PROFILE:
{json.dumps(compact_profile, ensure_ascii=False)}
"""

    try:
        r = client.chat.completions.create(
            model=character_model,
            temperature=0.35,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fictional-character personality "
                        "matching engine. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        text = r.choices[0].message.content or ""

        data = clean_json(text)

        return normalize_character(data, profile)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Character discovery failed: {e}"
        )

# ============================================================
# DATABASE SAVE
# ============================================================

def save_game_result(db: Session, character: dict):

    character_name = character.get("character_name")
    universe = character.get("universe")
    image_url = character.get("image_url")

    if not character_name or not universe:
        raise ValueError("Character name or universe is missing")

    # ---------------------------------------------------------
    # STEP 1: Check if this character already exists
    # ---------------------------------------------------------

    db_character = (
        db.query(Character)
        .filter(
            Character.character_name == character_name,
            Character.universe == universe
        )
        .first()
    )

    # ---------------------------------------------------------
    # STEP 2: Character already exists
    # ---------------------------------------------------------

    if db_character is not None:

        print(
            f"[DATABASE] Existing character found: "
            f"{character_name} | {universe}"
        )

        # IMPORTANT:
        # Reuse the image already stored in PostgreSQL.
        # Do NOT search AniList/Jikan/TMDB/Wikipedia again.

        character["image_url"] = db_character.image_url

    # ---------------------------------------------------------
    # STEP 3: Character is completely new
    # ---------------------------------------------------------

    else:

        print(
            f"[DATABASE] New character: "
            f"{character_name} | {universe}"
        )

        # Only a NEW character needs an image search.
        if not image_url:
            image_url = DEFAULT_CHARACTER_IMAGE

        db_character = Character(
            character_name=character_name,
            universe=universe,
            image_url=image_url
        )

        db.add(db_character)
        db.flush()

        print(
            f"[DATABASE] Image stored: {image_url}"
        )

        character["image_url"] = image_url

    # ---------------------------------------------------------
    # STEP 4: Save this player's result
    # ---------------------------------------------------------

    result = GameResult(
        character_id=db_character.id,
        match_percentage=character.get("match_percentage"),
        stats=character.get("stats", {})
    )

    db.add(result)

    db.commit()
    db.refresh(result)

    return result

# ============================================================
# QUESTIONS API
# ============================================================

@app.get("/api/questions")
def questions():

    return {
        "questions": QUESTIONS
    }


# ============================================================
# NORMAL ANALYZE API
# ============================================================

@app.post("/api/analyze")
async def analyze(
    req: GameRequest,
    db: Session = Depends(get_db)
):

    if len(req.answers) != 10:
        raise HTTPException(
            400,
            "Exactly 10 answers are required."
        )

    if any(len(a.strip()) < 12 for a in req.answers):
        raise HTTPException(
            400,
            "Each answer should be at least 12 characters."
        )

    # Analyze answers
    profile = await asyncio.to_thread(
        analyze_answers_sync,
        req.answers
    )

    # Find character
    characters = await asyncio.to_thread(
        discover_characters_sync,
        profile
    )

    character_name = characters["character_name"]
    universe = characters["universe"]

    # ---------------------------------------------------------
    # CHECK DATABASE FIRST
    # ---------------------------------------------------------

    db_character = (
        db.query(Character)
        .filter(
            Character.character_name == character_name,
            Character.universe == universe
        )
        .first()
    )

    if db_character is not None:

        print(
            f"[IMAGE] Using stored image for "
            f"{character_name} | {universe}"
        )

        characters["image_url"] = db_character.image_url

    else:

        print(
            f"[IMAGE] New character: "
            f"{character_name} | {universe}"
        )

        characters["image_url"] = await asyncio.to_thread(
            get_character_image,
            character_name,
            universe
        )

    # Save result
    try:

        save_game_result(
            db,
            characters
        )

    except Exception as e:

        db.rollback()

        print(
            f"[DATABASE ERROR] {e}"
        )

    return {
        "profile": profile,
        "character": characters,
        "characters": {
            "candidates": [characters]
        }
    }

# ============================================================
# STREAMING ANALYZE API
# ============================================================

@app.post("/api/analyze/stream")
async def analyze_stream(
    request: GameRequest,
    db: Session = Depends(get_db)
):

    if len(request.answers) != 10:
        raise HTTPException(
            400,
            "Exactly 10 answers are required."
        )

    if any(len(a.strip()) < 12 for a in request.answers):
        raise HTTPException(
            400,
            "Each answer should be at least 12 characters."
        )

    async def events():

        try:

            # -------------------------------------------------
            # 1. Analyze the player's answers
            # -------------------------------------------------

            yield json.dumps({
                "type": "progress",
                "message": "Understanding all 10 answers...",
                "percent": 12
            }) + "\n"

            profile = await asyncio.to_thread(
                analyze_answers_sync,
                request.answers
            )

            # -------------------------------------------------
            # 2. Discover the fictional character
            # -------------------------------------------------

            yield json.dumps({
                "type": "progress",
                "message": "Discovering your closest fictional character...",
                "percent": 56
            }) + "\n"

            characters = await asyncio.to_thread(
                discover_characters_sync,
                profile
            )

            character_name = characters["character_name"]
            universe = characters["universe"]

            # -------------------------------------------------
            # 3. CHECK DATABASE BEFORE IMAGE SEARCH
            # -------------------------------------------------

            yield json.dumps({
                "type": "progress",
                "message": "Checking character archive...",
                "percent": 70
            }) + "\n"

            db_character = (
                db.query(Character)
                .filter(
                    Character.character_name == character_name,
                    Character.universe == universe
                )
                .first()
            )

            # -------------------------------------------------
            # 4A. CHARACTER ALREADY EXISTS
            # -------------------------------------------------

            if db_character is not None:

                print(
                    f"[IMAGE] Using stored image for "
                    f"{character_name} | {universe}"
                )

                characters["image_url"] = (
                    db_character.image_url
                )

            # -------------------------------------------------
            # 4B. NEW CHARACTER
            # -------------------------------------------------

            else:

                print(
                    f"[IMAGE] New character detected: "
                    f"{character_name} | {universe}"
                )

                yield json.dumps({
                    "type": "progress",
                    "message": "Finding your character image...",
                    "percent": 78
                }) + "\n"

                image_url = await asyncio.to_thread(
                    get_character_image,
                    character_name,
                    universe
                )

                characters["image_url"] = image_url

            # -------------------------------------------------
            # 5. SAVE RESULT
            # -------------------------------------------------

            save_game_result(
                db,
                characters
            )

            # -------------------------------------------------
            # 6. Finish
            # -------------------------------------------------

            yield json.dumps({
                "type": "progress",
                "message": "Preparing your results...",
                "percent": 92
            }) + "\n"

            yield json.dumps({
                "type": "result",
                "profile": profile,
                "character": characters,
                "characters": {
                    "candidates": [characters]
                },
                "percent": 100
            }) + "\n"

        except Exception as e:

            print(
                f"[STREAM ERROR] {type(e).__name__}: {e}"
            )

            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson"
    )


# ============================================================
# STATISTICS API
# ============================================================

@app.get("/api/statistics")
def get_statistics(
    db: Session = Depends(get_db)
):

    total_players = (
        db.query(GameResult).count()
    )

    character_results = (

        db.query(

            Character.character_name,

            Character.universe,

            Character.image_url,

            func.count(
                GameResult.id
            ).label("count")

        )

        .join(
            GameResult,
            GameResult.character_id
            == Character.id
        )

        .group_by(

            Character.id,

            Character.character_name,

            Character.universe,

            Character.image_url
        )

        .order_by(
            func.count(
                GameResult.id
            ).desc()
        )

        .all()
    )

    characters = []

    for row in character_results:

        percentage = (

            (
                row.count
                / total_players
            ) * 100

            if total_players > 0
            else 0
        )

        characters.append({

            "character_name":
                row.character_name,

            "universe":
                row.universe,

            "image_url":
                row.image_url,

            "count":
                row.count,

            "percentage":
                round(
                    percentage,
                    1
                )
        })

    return {

        "total_players":
            total_players,

        "characters":
            characters
    }


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)