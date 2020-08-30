import discord
from discord.ext import commands
import asyncio # To get the exception
from tweetpull import pull
from twitter import twitter
from discord import Game
from graph import initHour, initDaily, tickerInfo, trendHour, trendDay, sFinancials, sDividends

bot = commands.Bot(command_prefix= '!')

@bot.event
async def on_ready():
    await bot.change_presence(activity=Game(name="with ur mom"))
    #watching preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="a movie"))
    #listening preset
    #await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="a song"))
    print('Bot is ready.')

@client.command(aliases=["quit"])
@commands.has_permissions(administrator=True)
async def close(ctx):
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
    # await ctx.send("Volume: " + str(volume) + '\n' +
    #                 "Average Volume: " + str(averageVolume) + '\n' +
    #                 "Average 10 Day Volume: " + str(averageVolume10Days) + '\n' +
    #                 "200 Day Average: " + str(twoHundredDayAverage) + '\n' +
    #                 "Shares Short: " + str(sharesShort) + '\n' +
    #                 "Short Ratio: " + str(shortRatio) + '\n' +
    #                 "Float Shares: " + str(floatShares) + '\n' +
    #                 "% Institutional Holdings: " + str(heldPercentInstitutions)+ '\n' +
    #                 "% Insider Holdings: " + str(heldPercentInsiders))

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
    # await ctx.send("Trailing PE: " + str(trailingPE) + '\n' +
    #                 "Profit Margins: " + str(profitMargins) + '\n' +
    #                 "PEG Ratio: " + str(pegRatio) + '\n' +
    #                 "Quarterly Earnings Growth: " + str(earningsQuarterlyGrowth) + '\n' +
    #                 "Price-Book: " + str(priceToBook) + '\n' +
    #                 "Market Cap: " + str(marketCap)+ '\n' +
    #                 "Forward PE: " + str(forwardPE)+ '\n' +
    #                 "Enterprise-Ebita: " + str(enterpriseToEbita)+ '\n' +
    #                 "Forward EPS: " + str(forwardEPS)+ '\n' +
    #                 "Shares Outstanding: " + str(sharesOutstanding)+ '\n' +
    #                 "Book Value: " + str(bookValue)+ '\n' +
    #                 "Trailing EPS: " + str(trailingEPS)+ '\n' +
    #                 "Enterprise Value: " + str(enterpriseValue))


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
    # await ctx.send("Payout Ratio: " + str(payoutRatio) + '\n' +
    #                 "Dividend Rate: " + str(dividendRate) + '\n' +
    #                 "Dividend Yield: " + str(dividendYield) + '\n' +
    #                 "Five Year Average Dividend Yield: " + str(fiveYearAvgDividendYield) + '\n' +
    #                 "Trailing Annual Dividend Yield: " + str(trailingAnnualDividendYield))



@bot.command(pass_context=True)
async def graphHour(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On 1 Hour Frequency...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initHour(message)
    

@bot.command(pass_context=True)
async def graphDay(ctx,*,message):
    #mention = ctx.message.author.mention
    #channel = bot.get_channel(748394087522238495)
    await ctx.send("Displaying Graph for Ticker: " + message + " On Daily Frequency...")
    #await channel.send(f"Displaying results for Sentiment Analysis ran by {mention}!")
    initDaily(message)

@bot.command(pass_context=True)
async def trendlineHour(ctx,*,message):
    await ctx.send("Displaying Trendlines for Ticker: " + message + " On 1 Hour Frequency...")
    trendHour(message)

@bot.command(pass_context=True)
async def trendlineDay(ctx,*,message):
    await ctx.send("Displaying Trendlines for Ticker: " + message + " On Daily Frequency...")
    trendDay(message)
    

@bot.command()
async def clear(ctx, amount=5):
    await ctx.channel.purge(limit=amount)

@bot.command()
async def commandlist(ctx):
    embed = discord.Embed(title="Commands Help", description="Some useful commands")
    embed.add_field(name="!run (ticker)", value="Runs Twitter Sentiment")
    embed.add_field(name="!stockStats (ticker)", value="Shows Stock Statistics Based on Given Ticker")
    embed.add_field(name="!stockFinancials (ticker)", value="Shows Stock Financials Based on Given Ticker")
    embed.add_field(name="!stockDividends (ticker)", value="Shows Stock Dividends Based on Given Ticker")
    embed.add_field(name="!graphHour (ticker)", value="1 Hour Graph of the Past Month")
    embed.add_field(name="!graphDay (ticker)", value="Daily Graph of the Past 6 Months")
    embed.add_field(name="!trendlineHour (ticker)", value="Draws Trendlines on a 1 Hour Graph of the Past 2 Weeks")
    embed.add_field(name="!trendlineDay (ticker)", value="Draws Trendlines on a Daily Graph of the Past 6 Months")
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