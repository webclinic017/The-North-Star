from finvizfinance.news import News
import autosentiment as at
from dhooks import Webhook, File

hook = Webhook("https://discordapp.com/api/webhooks/755303399712358430/HW84bs4k2TJAIJXPKi2ZjLIgBaMsYwRmFwm4mYmhDhNTdH6aRsAZd_uhd3d4WooGrH4-")
with open('news.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='NewsBot', avatar=img)

negHook = Webhook("https://discordapp.com/api/webhooks/755305497237913710/3u4WEd66E_dEymW69fbA3y9h215mRgZHFoyTgwm63vCe9fUGy0X1Ke_0Tcs2Ry8fEDN7")
with open('news.png', 'r+b') as f:
    img = f.read()  # bytes

negHook.modify(name='NewsBot', avatar=img)
#finvizfinance
fnews = News()
all_news = fnews.getNews()

all_news['news']['sentiment'] = 0
all_news['news']['sentiment'] =at.analysis_ternary(all_news['news']['Title'])
sentiment = all_news['news']['sentiment'][:25]

for i in range(25):
    if sentiment[i] == 1.0:
        hook.send(all_news['news']['Link'][i])
    elif sentiment[i] == -1.0:
        negHook.send(all_news['news']['Link'][i])
