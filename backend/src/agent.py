import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    room_io,
    tokenize,
)
from livekit.plugins import (
    deepgram,
    google,
    murf,
    noise_cancellation,
    silero,
)

# NOTE: MultilingualModel import removed on purpose (it used too much RAM on Railway)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "shiksha_memory.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            current_level TEXT,
            topics_covered TEXT,
            last_interaction TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS call_logs (
            call_id TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            outcome TEXT NOT NULL,
            reason TEXT
        )
        """
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized: %s", DB_PATH)


init_db()


# ============================================================
# CLEANUP STUCK CALLS
# ============================================================


def cleanup_stuck_calls():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE call_logs
        SET
            outcome = 'failed',
            reason = 'interrupted - agent restarted',
            end_time = ?
        WHERE outcome = 'in_progress'
        """,
        (datetime.now().isoformat(),),
    )

    affected = cursor.rowcount
    conn.commit()
    conn.close()

    if affected:
        logger.info("Marked %s stuck call(s) as failed.", affected)


cleanup_stuck_calls()


# ============================================================
# CALL ANALYTICS
# ============================================================


def log_call_start(call_id: str, channel: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO call_logs (
            call_id, channel, start_time, end_time, outcome, reason
        )
        VALUES (?, ?, ?, NULL, 'in_progress', NULL)
        """,
        (call_id, channel, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def log_call_end(call_id: str, outcome: str, reason: str = ""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE call_logs
        SET end_time = ?, outcome = ?, reason = ?
        WHERE call_id = ?
        """,
        (datetime.now().isoformat(), outcome, reason, call_id),
    )
    conn.commit()
    conn.close()


# ============================================================
# SHARED CALL STATE
# ============================================================


class CallState:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.exercise_completed = False
        self.escalation_created = False


# ============================================================
# NEO - GENERAL PURPOSE AI VOICE ASSISTANT PROMPT
# ============================================================

SYSTEM_PROMPT = """
IDENTITY:
Your name is Neo. You are a friendly, capable and versatile AI voice assistant that people talk to through a mobile app.
You were created by two developers, Naman and Mayuresh.
You help the user with anything they need, whether it is learning a new concept, discussing technology, chatting casually, or solving problems.

ABOUT YOURSELF (very important):
When the user asks things like "who are you", "what can you do", "who made you", "who created you", "introduce yourself" or "tell me about yourself", answer in about 4 to 5 short spoken sentences, in this order:
1. Say that you are Neo, an AI voice assistant.
2. Say that you were built by Naman and Mayuresh.
3. Briefly say what you can do: talk with the user in real time about almost any topic, answer questions and explain things in simple words, help the user practice for interviews, help improve their English by speaking, and run quick quizzes and math practice.
4. You may mention that more features, like interview practice based on the user's resume and files, are coming soon.
5. End by asking what the user would like to do.
If the user only asks who made you, answer briefly that Naman and Mayuresh built you, and then offer your help.
Never say that you were made by Google, OpenAI or any other company. If asked which AI model you use, say that you are Neo, built by Naman and Mayuresh, and that you do not go into technical details.

ROLE & CAPABILITIES:
- Open-Ended Conversation: You can discuss any topic (science, daily life, movies, history, etc.) naturally and intelligently.
- Chit-Chat: Be conversational, empathetic, and responsive to the user's mood. Feel free to joke, brainstorm, or just chat.
- Education & Tutoring: If the user specifically wants to study, you can explain complex topics simply, or use your tools to generate quizzes and math problems.
- Interview Practice: If the user wants to practice for an interview, ask one interview question at a time, listen to the answer, and give short, helpful feedback.
- English Practice: If the user wants to improve their English, chat with them, gently correct their mistakes, and encourage them.

LANGUAGE & TONE:
- Understand whatever the user speaks (English, Hindi, Hinglish).
- Respond in clear, natural conversational English (to ensure the Text-to-Speech engine pronounces it perfectly).
- Keep your answers concise, engaging, and speech-optimized (usually 1-3 sentences per turn). Do not give long monologues unless asked. The only exception is your introduction, which can be 4 to 5 sentences.
- Avoid using emojis, markdown, or bullet points in your speech.
"""


# ============================================================
# ALL-PURPOSE ASSISTANT AGENT
# ============================================================


class GeneralAIAssistant(Agent):
    def __init__(self, state: CallState):
        super().__init__(instructions=SYSTEM_PROMPT)
        self.state = state

    # ========================================================
    # EDUCATIONAL QUIZ TOOL
    # ========================================================
    @function_tool(
        description="Fetch an educational trivia/quiz question. Call only if the user explicitly wants to play a quiz or test their knowledge."
    )
    async def fetch_educational_quiz(
        self,
        context: RunContext,
        subject: Annotated[str, "Requested subject: math or general"],
    ) -> str:
        subject = (subject or "general").strip().lower()
        category = 19 if "math" in subject else 9
        api_url = (
            f"https://opentdb.com/api.php?amount=1&category={category}&type=multiple"
        )

        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(api_url) as response,
            ):
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                data = await response.json()

            item = data.get("results", [])[0]
            options = [*item.get("incorrect_answers", []), item.get("correct_answer")]

            return json.dumps(
                {
                    "status": "success",
                    "question": item.get("question"),
                    "correct_answer": item.get("correct_answer"),
                    "options": options,
                },
                ensure_ascii=False,
            )

        except Exception as error:
            logger.warning("QUIZ FETCH FALLBACK USED: %s", error)
            return json.dumps(
                {
                    "status": "fallback",
                    "question": "What is the capital of Australia?",
                    "correct_answer": "Canberra",
                    "options": ["Sydney", "Melbourne", "Canberra", "Perth"],
                }
            )

    # ========================================================
    # MATH PROBLEM GENERATOR TOOL
    # ========================================================
    @function_tool(
        description="Generate a math practice question. Call only if the user wants to practice math."
    )
    async def generate_math_problem(
        self,
        context: RunContext,
        level: Annotated[
            str, "Student skill level: beginner, intermediate, or advanced"
        ],
    ) -> str:
        level = (level or "beginner").strip().lower()
        if "advanced" in level:
            question = "What is 15 multiplied by 24?"
        elif "intermediate" in level:
            question = "If 3x plus 7 equals 22, what is the value of x?"
        else:
            question = "What is 15 multiplied by 6?"

        return json.dumps(
            {
                "status": "success",
                "subject": "mathematics",
                "level": level,
                "question": question,
            }
        )

    # ========================================================
    # MARK EXERCISE COMPLETE TOOL
    # ========================================================
    @function_tool(
        description="Mark the current exercise as completed once the user answers correctly."
    )
    async def mark_exercise_complete(self, context: RunContext) -> str:
        self.state.exercise_completed = True
        return "Exercise completion recorded successfully."

    # ========================================================
    # PROFILE & ESCALATION TOOLS
    # ========================================================
    @function_tool(description="Look up a user's profile by name or ID.")
    async def get_user_profile(
        self, context: RunContext, user_id: Annotated[str, "Unique user name or ID"]
    ) -> str:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE user_id = ?", (user_id.lower(),))
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.dumps(
                {
                    "user_id": row[0],
                    "name": row[1],
                    "current_level": row[2],
                    "topics_covered": row[3],
                    "last_interaction": row[4],
                }
            )
        return json.dumps({"status": "not_found"})

    @function_tool(description="Save or update a user's profile.")
    async def save_user_profile(
        self,
        context: RunContext,
        user_id: Annotated[str, "User ID"],
        name: Annotated[str, "Name"],
        current_level: Annotated[str, "Level"],
        topics_covered: Annotated[str, "Topics"],
    ) -> str:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?)",
            (
                user_id.lower(),
                name,
                current_level,
                topics_covered,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return "Profile saved successfully."


# ============================================================
# LIVEKIT SERVER & SESSION SETUP
# ============================================================

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# IMPORTANT: this name must match the agent name your frontend sends
@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    call_id = uuid.uuid4().hex[:8]
    state = CallState(call_id=call_id)

    channel = "browser"
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            if metadata.get("phone_number"):
                channel = "sip"
        except Exception:
            logger.warning("Could not parse job metadata")

    log_call_start(call_id, channel)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=google.LLM(model="gemini-3.5-flash-lite"),
        tts=murf.TTS(
            voice="Anisha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # Lightweight: uses Silero VAD only (no big turn-detector model)
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    assistant = GeneralAIAssistant(state=state)

    async def on_shutdown():
        log_call_end(call_id, "completed", "Call ended by user or system.")

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Phone (SIP) calls -> BVCTelephony, browser/app -> BVC
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()

    # Open-ended, friendly greeting
    await session.say(
        "Hi there! I am Neo, your AI assistant. How can I help you today?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
