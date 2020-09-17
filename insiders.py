from finvizfinance.insider import Insider
import dataframe_image as dfi
from dhooks import Webhook, Embed, File
import os
import seaborn as sns
import pandas as pd

hook = Webhook("https://discordapp.com/api/webhooks/755482527451578428/V6YiRb36KcQTcqlkTyk3FjMQLNYi6UNpu84doNOaBKCy2trm_On9yF__RS4prkuIzG_y")
with open('insider.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='InsiderBot', avatar=img)


finsider = Insider(option='top owner trade')
finsider2 = Insider(option='top week')
# option: latest, top week, top owner trade
# default: latest

insider_trader = finsider.getInsider()
df = insider_trader
del df['SEC Form 4']
del df['Relationship']
del df['#Shares Total']
df = df[['Date', 'Ticker', 'Owner', 'Transaction', 'Cost', '#Shares', 'Value ($)']]
cm = sns.light_palette("#F21A1A", as_cmap=True)
df_styled = df.style.background_gradient(cmap=cm)

df_styled = df_styled.hide_index()
dfi.export(df_styled,"INSIDERS.png")
discord_pic = File("INSIDERS.png")
hook.send("Top 10% Owners of Recent Trading Week",file=discord_pic)
os.remove("INSIDERS.png")

insider_trader2 = finsider2.getInsider()
dfb = insider_trader2
del dfb['SEC Form 4']
del dfb['Relationship']
del dfb['#Shares Total']
dfb = dfb[['Date', 'Ticker', 'Owner', 'Transaction', 'Cost', '#Shares', 'Value ($)']]
dfb = dfb[dfb['Value ($)'] > 10000000]
cm = sns.light_palette("#A80E34", as_cmap=True)
dfb_styled = dfb.style.background_gradient(cmap=cm)

dfb_styled = dfb_styled.hide_index()
dfi.export(dfb_styled,"INSIDERS2.png")
discord_pic = File("INSIDERS2.png")
hook.send("Top Owners of Recent Trading Week (Value Over $10 Million)",file=discord_pic)
os.remove("INSIDERS2.png")

#print(df)