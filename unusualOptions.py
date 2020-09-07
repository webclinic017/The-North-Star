from barchart import UOA
import pandas as pd
import dataframe_image as dfi
from dhooks import Webhook, File
import datetime as dt
import os



hook = Webhook("https://discordapp.com/api/webhooks/750589088209436712/mPrkem3sHTuyVW7EWZ5aQn5mfzvU0JJgrg2HobnyEqtjUH5P9q6Pr8oLKAn6RjJE8iw1")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)


def unusualOptions():
    unusual = UOA()
    unusual.to_csv()
    file_name = dt.datetime.now().strftime("%m%d%Y") + '_UOA.csv'
    df = pd.read_csv(file_name, nrows=25)
    df2 = pd.read_csv(file_name, skiprows=25, nrows=25)
    df.reset_index()
    df2.reset_index()
    # df=df.drop(['index'],axis=1)
    # df2=df2.drop(['index'],axis=1)
    df_styled = df.style.background_gradient()
    df_styled2 = df2.style.background_gradient()
    dfi.export(df_styled,"uoa.png") #adding a gradient based on values in cell
    dfi.export(df_styled2,"uoa2.png") #adding a gradient based on values in cell

    discord_pic = File('uoa.png')
    discord_pic2 = File('uoa2.png')
    hook.send("Top 50 Unusual Options of the Day",file=discord_pic)
    hook.send(file=discord_pic2)
    os.remove(file_name)
    os.remove('uoa.png')
    os.remove('uoa2.png')

unusualOptions()
