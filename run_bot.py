"""Run a Discord bot that does document Q&A using Modal and LangChain."""
import argparse
import asyncio
import logging
import os

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv
import requests

load_dotenv()

MODAL_USER_NAME = os.environ["MODAL_USER_NAME"]
BACKEND_URL = f"https://{MODAL_USER_NAME}--llm-qna-hook.modal.run"

guild_ids = {"dev": 1070516629328363591}

START, END = "\033[1;36m", "\033[0m"


async def runner(query, request_id=None):
    payload = {"query": query}
    if request_id:
        payload["request_id"] = request_id
    async with aiohttp.ClientSession() as session:
        async with session.get(url=BACKEND_URL, params=payload) as response:
            assert response.status == 200, response
            json_content = await response.json()
    return json_content["answer"]


def main(auth, guilds):
    # Discord auth requires statement of "intents"
    #  we start with default behaviors
    intents = discord.Intents.default()
    #  and add reading messages
    intents.message_content = True

    bot = commands.Bot(intents=intents, guilds=guilds)

    rating_emojis = {
        "👍": "if the response was helpful", "👎": "if the response was not helpful"
    }

    emoji_reaction_text = " or ".join(f"react with {emoji} {reason}" for emoji, reason in rating_emojis.items())
    emoji_reaction_text = emoji_reaction_text.capitalize() + "."

    @bot.event
    async def on_ready():
       pretty_log(f"{bot.user} is ready and online!")

    response_fmt = \
    """{mention} asked: {question}

    Here's my best guess at an answer, with sources so you can follow up:

    {answer}

    Emoji react to let us know how we're doing!

    """

    response_fmt += emoji_reaction_text

    # add our command
    @bot.slash_command(name="ask")
    @discord.option(
    "question",
    str,
    description="A question about LLMs."
    )
    async def answer(ctx, question: str):
        """Answers questions about LLMs."""

        respondent = ctx.author

        pretty_log(f"responding to question \"{question}\"")
        await ctx.defer(ephemeral=False, invisible=False)
        original_message = await ctx.interaction.original_response()
        message_id = original_message.id
        answer = await runner(question, request_id=message_id)
        answer.strip()
        await ctx.respond(response_fmt.format(mention=respondent.mention, question=question, answer=answer))  # respond
        for emoji in rating_emojis:
            await original_message.add_reaction(emoji)
            await asyncio.sleep(0.25)


    @bot.slash_command()
    async def health(ctx):
        "Supports a Discord bot version of a liveness probe."
        pretty_log(f"inside healthcheck")
        await ctx.respond("200 more like 💯 mirite")

    bot.run(auth)


def pretty_log(str):
    print(f"{START}🤖: {str}{END}")


if __name__ == "__main__":
    guilds = [guild_ids["dev"]]
    auth = os.environ["DISCORD_AUTH_DEV"]
    main(auth=auth, guilds=guilds)
