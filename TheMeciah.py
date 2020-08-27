import discord
from discord.ext import commands
import asyncio # To get the exception
from tweetpull import pull
from twitter import twitter
from discord import Game
from graph import initHour, initDaily, tickerInfo

bot = commands.Bot(command_prefix= '!')

@bot.event
async def on_ready():
    await bot.change_presence(activity=Game(name="with humans"))
    #watching preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="a movie"))
    #listening preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="a song"))
    print('Bot is ready.')



@bot.command(pass_context=True)
async def run(ctx,*,message):
    mention = ctx.message.author.mention
    channel = bot.get_channel(746808423185776740)
    
    await ctx.send("Running bot for keywords: " + message+ "...")
    pull(message)
    await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    twitter(message)


@bot.command(pass_context=True)
async def stockStats(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Statistics for Ticker: " + message)
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    averageVolume, twoHundredDayAverage, averageVolume10Days, heldPercentInstitutions, heldPercentInsiders, volume, sharesShort, shortRatio, floatShares= tickerInfo(message)
    await ctx.send("Volume: " + str(volume) + '\n' +
                    "Average Volume: " + str(averageVolume) + '\n' +
                    "Average 10 Day Volume: " + str(averageVolume10Days) + '\n' +
                    "200 Day Average: " + str(twoHundredDayAverage) + '\n' +
                    "Shares Short: " + str(sharesShort) + '\n' +
                    "Short Ratio: " + str(shortRatio) + '\n' +
                    "Float Shares: " + str(floatShares) + '\n' +
                    "% Institutional Holdings: " + str(heldPercentInstitutions)+ '\n' +
                    "% Insider Holdings: " + str(heldPercentInsiders))

@bot.command(pass_context=True)
async def stockFinancials(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Financials for Ticker: " + message)
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    trailingPE, marketCap, forwardPE, enterpriseToEbita, forwardEPS, sharesOutstanding, bookValue, trailingEPS, enterpriseValue  = stockFinancials(message)
    await ctx.send("Trailing PE: " + str(trailingPE) + '\n' +
                    "Profit Margins: " + str(profitMargins) + '\n' +
                    "PEG Ratio: " + str(pegRatio) + '\n' +
                    "Quarterly Earnings Growth: " + str(earningsQuarterlyGrowth) + '\n' +
                    "Price-Book: " + str(priceToBook) + '\n' +
                    "Market Cap: " + str(marketCap)+ '\n' +
                    "Forward PE: " + str(forwardPE)+ '\n' +
                    "Enterprise-Ebita: " + str(enterpriseToEbita)+ '\n' +
                    "Forward EPS: " + str(forwardEPS)+ '\n' +
                    "Shares Outstanding: " + str(sharesOutstanding)+ '\n' +
                    "Book Value: " + str(bookValue)+ '\n' +
                    "Trailing EPS: " + str(trailingEPS)+ '\n' +
                    "Enterprise Value: " + str(enterpriseValue))


@bot.command(pass_context=True)
async def stockDividends(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Financials for Ticker: " + message)
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    trailingAnnualDividendYield, payoutRatio, dividendYield, dividendRate, fiveYearAvgDividendYield = stockDividends(message)
    await ctx.send("Payout Ratio: " + str(averageVolume) + '\n' +
                    "Dividend Rate: " + str(dividendRate) + '\n' +
                    "Dividend Yield: " + str(dividendYield) + '\n' +
                    "Five Year Average Dividend Yield: " + str(fiveYearAvgDividendYield) + '\n' +
                    "Trailing Annual Dividend Yield: " + str(trailingAnnualDividendYield))



@bot.command(pass_context=True)
async def graphHour(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initHour(message)
    

@bot.command(pass_context=True)
async def graphDaily(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initDaily(message)
    

@bot.command()
async def clear(ctx, amount=5):
    await ctx.channel.purge(limit=amount)

@bot.command()
async def commandlist(ctx):
    embed = discord.Embed(title="Commands Help", description="Some useful commands")
    embed.add_field(name="!run (keyword)", value="Runs Twitter Sentiment")
    embed.add_field(name="!clear (number)", value="Clears Previous Lines based on number. Defualt is 5")
    await ctx.send(content=None, embed=embed)

# @bot.command(pass_context=True)
# async def name(ctx,*,message):
#     username = ctx.message.author.display_name
#     mention = ctx.message.author.mention
#     channel = bot.get_channel(746808423185776740)
#     await channel.send(message)
    #await ctx.send(f"hey {mention}, you're great!")

bot.run('NzQ2ODE3MDkxNTAzNTIxODYy.X0F1nQ.NFTXuu6aCSVL3MfbBh2WUzj7_4s')