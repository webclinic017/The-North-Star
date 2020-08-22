import discord
from discord.ext import commands
import asyncio # To get the exception
from tweetpull import pull
from twitter import twitter

bot = commands.Bot(command_prefix= '!')

@bot.event
async def on_ready():
    print('Bot is ready.')



@bot.command(pass_context=True)
async def run(ctx,*,message):
    await ctx.send("Running bot for keywords: " + message+ "...")
    pull(message)
    twitter(message)
    

# @bot.command()
# async def test(ctx, arg):
#     await ctx.send(arg)

bot.run('NzQ2ODE3MDkxNTAzNTIxODYy.X0F1nQ.NFTXuu6aCSVL3MfbBh2WUzj7_4s')