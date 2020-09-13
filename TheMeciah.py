import discord
from discord.ext import commands
import asyncio # To get the exception
from tweetpull import pull
from twitter import twitter
from discord import Game
from graph import initHour, initDaily, tickerInfo, trendHour, trendDay, sFinancials, sDividends, graphDailyCH, graphhrCH, graphDailyV, graphhrV, graphDailyMF, graphhrMF, graphDailyFT, graphhrFT

bot = commands.Bot(command_prefix= '!')

@bot.event
async def on_ready():
    await bot.change_presence(activity=Game(name="with ur mom"))
    #watching preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="a movie"))
    #listening preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="a song"))
    print('Bot is ready.')

@bot.command(aliases=["quit"])
async def close(ctx):
    if ctx.message.author.id == (128313988671995904):
        await bot.close()

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
    averageVolume, twoHundredDayAverage, averageVolume10Days, heldPercentInstitutions, heldPercentInsiders, volume, sharesShort, shortRatio, floatShares = tickerInfo(message)
    embed = discord.Embed(title="Dividends Information", description="")
    embed.add_field(name="Volume", value=str(volume))
    embed.add_field(name="Average Volume", value=str(averageVolume))
    embed.add_field(name="Average 10 Day Volume", value=str(averageVolume10Days))
    embed.add_field(name="200 Day Average", value=str(twoHundredDayAverage))
    embed.add_field(name="Shares Short", value=str(sharesShort))
    embed.add_field(name="Short Ratio", value=str(shortRatio))
    embed.add_field(name="Float Shares", value=str(floatShares))
    embed.add_field(name="% Institutional Holdings", value=str(heldPercentInstitutions))
    embed.add_field(name="% Insider Holdings", value=str(heldPercentInsiders))
    await ctx.send(content=None, embed=embed)


@bot.command(pass_context=True)
async def stockFinancials(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Financials for Ticker: " + message)
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    trailingPE, marketCap, earningsQuarterlyGrowth, priceToBook, pegRatio, profitMargins, forwardPE, enterpriseToEbita, forwardEPS, sharesOutstanding, bookValue, trailingEPS, enterpriseValue = sFinancials(message)
    embed = discord.Embed(title="Financial Information", description="")
    embed.add_field(name="Trailing PE", value=str(trailingPE))
    embed.add_field(name="Market Cap", value=str(marketCap))
    embed.add_field(name="Quarterly Earnings Growth", value=str(earningsQuarterlyGrowth))
    embed.add_field(name="Price-Book", value=str(priceToBook))
    embed.add_field(name="PEG Ratio", value=str(pegRatio))
    embed.add_field(name="Profit Margins", value=str(profitMargins))
    embed.add_field(name="Forward PE", value=str(forwardPE))
    embed.add_field(name="Enterprise-Ebita", value=str(enterpriseToEbita))
    embed.add_field(name="Forward EPS", value=str(forwardEPS))
    embed.add_field(name="Shares Outstanding", value=str(sharesOutstanding))
    embed.add_field(name="Book Value", value=str(bookValue))
    embed.add_field(name="Trailing EPS", value=str(trailingEPS))
    embed.add_field(name="Enterprise Value", value=str(enterpriseValue))
    await ctx.send(content=None, embed=embed)

@bot.command(pass_context=True)
async def stockDividends(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Dividends for Ticker: " + message)
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    trailingAnnualDividendYield, payoutRatio, dividendYield, dividendRate, fiveYearAvgDividendYield = sDividends(message)
    embed = discord.Embed(title="Dividends Information", description="")
    embed.add_field(name="Payout Ratio", value=str(payoutRatio))
    embed.add_field(name="Dividend Rate", value=str(dividendRate))
    embed.add_field(name="Dividend Yield", value=str(dividendYield))
    embed.add_field(name="Five Year Average Dividend Yield", value=str(fiveYearAvgDividendYield))
    embed.add_field(name="Trailing Annual Dividend Yield", value=str(trailingAnnualDividendYield))
    await ctx.send(content=None, embed=embed)


@bot.command(pass_context=True)
async def graphHour(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initHour(message)
    

@bot.command(pass_context=True)
async def graphDay(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initDaily(message)

@bot.command(pass_context=True)
async def graphDayCH(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    graphDailyCH(message)

@bot.command(pass_context=True)
async def graphHourCH(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    graphhrCH(message)

@bot.command(pass_context=True)
async def graphDayV(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    graphDailyV(message)

@bot.command(pass_context=True)
async def graphHourV(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency in Channel (stock-data)...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    graphhrV(message)

@bot.command(pass_context=True)
async def graphDayMF(ctx,*,message):
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency in Channel (stock-data)...")
    graphDailyMF(message)

@bot.command(pass_context=True)
async def graphHourMF(ctx,*,message):
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency in Channel (stock-data)...")
    graphhrMF(message)

@bot.command(pass_context=True)
async def graphDayFT(ctx,*,message):
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency in Channel (stock-data)...")
    graphDailyFT(message)

@bot.command(pass_context=True)
async def graphHourFT(ctx,*,message):
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency in Channel (stock-data)...")
    graphhrFT(message)


@bot.command(pass_context=True)
async def trendlineHour(ctx,*,message):
    await ctx.send("Displaying Trendlines for Ticker: " + message + " On 1 Hour Frequency in Channel (trendlines)...")
    trendHour(message)

@bot.command(pass_context=True)
async def trendlineDay(ctx,*,message):
    await ctx.send("Displaying Trendlines for Ticker: " + message + " On Daily Frequency in Channel (trendlines)...")
    trendDay(message)


    

@bot.command()
async def clear(ctx, amount=5):
    if ctx.message.author.id == (128313988671995904) or ctx.message.author.id == (432210386100420617):
        await ctx.channel.purge(limit=amount)

@bot.command()
async def commandlist(ctx):
    embed = discord.Embed(title="Commands Help", description="Some useful commands for TheMeciah")
    embed.add_field(name="!run (ticker)", value="Runs Twitter Sentiment")
    embed.add_field(name="!stockStats (ticker)", value="Shows Stock Statistics Based on Given Ticker")
    embed.add_field(name="!stockFinancials (ticker)", value="Shows Stock Financials Based on Given Ticker")
    embed.add_field(name="!stockDividends (ticker)", value="Shows Stock Dividends Based on Given Ticker")
    embed.add_field(name="!graphHour (ticker)", value="1 Hour Graph of the Past Week")
    embed.add_field(name="!graphDay (ticker)", value="Daily Graph of the Past 6 Months")
    embed.add_field(name="!graphHourCH (ticker)", value="1 Hour Chaikin Graph of the Past Week")
    embed.add_field(name="!graphDayCH (ticker)", value="Daily Chaikin Graph of the Past 6 Months")
    embed.add_field(name="!graphHourV (ticker)", value="1 Hour Vortex Graph of the Past Week")
    embed.add_field(name="!graphDayV (ticker)", value="Daily Vortex Graph of the Past 6 Months")
    embed.add_field(name="!trendlineHour (ticker)", value="Draws Trendlines on a 1 Hour Graph of the Past 2 Weeks")
    embed.add_field(name="!trendlineDay (ticker)", value="Draws Trendlines on a Daily Graph of the Past 6 Months")
    embed.add_field(name="!graphHourMF (number)", value="1 Hour Chaikin Graph of the Past Week")
    embed.add_field(name="!graphDayMF (number)", value="Daily Money Flow Graph of the Past 6 Months")
    embed.add_field(name="!graphHourFT (number)", value="1 Hour Fisher Transform Graph of the Past Week")
    embed.add_field(name="!graphDayFT (number)", value="Daily Fisher Transform Graph of the Past 6 Months")
    await ctx.send(content=None, embed=embed)

# @bot.command(pass_context=True)
# async def name(ctx,*,message):
#     username = ctx.message.author.display_name
#     mention = ctx.message.author.mention
#     channel = bot.get_channel(746808423185776740)
#     await channel.send(message)
    #await ctx.send(f"hey {mention}, you're great!")

bot.run('NzQ2ODE3MDkxNTAzNTIxODYy.X0F1nQ.NFTXuu6aCSVL3MfbBh2WUzj7_4s')