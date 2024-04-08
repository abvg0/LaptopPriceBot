import telebot
from telebot import types

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


scrs = {
    "HD": {"count": 1049088, 'callback': '51'},
    "HD+": {"count": 1440000, 'callback': '52'},
    "FullHD": {"count": 2073600, 'callback': '53'},
    "2K": {"count": 2211840, 'callback': '54'},
    "4K UltraHD": {"count": 8294400, 'callback': '55'}
}

cpus = {
    "Intel Core i5": {"count": 2.3, 'callback': '37'},
    "Intel Core i7": {"count": 2.7, 'callback': '38'},
    "Intel Core i3": {"count": 2.0, 'callback': '39'},
    "Intel Core M": {"count": 1.2, 'callback': '40'},
    "AMD A12-Series": {"count": 2.5, 'callback': '41'},
    "AMD E-Series": {"count": 1.8, 'callback': '42'},
    "AMD A9-Series": {"count": 3.0, 'callback': '43'},
    "AMD A6-Series": {"count": 2.0, 'callback': '44'},
    "Intel Pentium": {"count": 2.1, 'callback': '45'},
    "AMD A4-Series": {"count": 2.2, 'callback': '46'},
    "Intel Celeron": {"count": 2.0, 'callback': '47'},
    "Samsung Cortex": {"count": 2.0, 'callback': '48'},
    "AMD FX": {"count": 2.1, 'callback': '49'},
    "Intel Xeon": {"count": 2.9, 'callback': '50'}
}

rams = {
    "4GB": {"count": 4, 'callback': '29'},
    "6GB": {"count": 6, 'callback': '30'},
    "8GB": {"count": 8, 'callback': '31'},
    "12GB": {"count": 12, 'callback': '32'},
    "16GB": {"count": 16, 'callback': '33'},
    "24GB": {"count": 24, 'callback': '34'},
    "32GB": {"count": 32, 'callback': '35'},
    "64GB": {"count": 64, 'callback': '36'}
}

memorys = {
    "128GB": {"count": 128, 'callback': '21'},
    "128GB + 1TB": {"count": 1152, 'callback': '22'},
    "256GB": {"count": 256, 'callback': '23'},
    "256GB + 1TB": {"count": 1280, 'callback': '24'},
    "512GB": {"count": 512, 'callback': '25'},
    "512GB + 1TB": {"count": 1536, 'callback': '26'},
    "1TB": {"count": 1024, 'callback': '27'},
    "1TB + 1TB": {"count": 2048, 'callback': '28'}
}
gpus = {
    'GTX 980 SLI': {'rang': 108, 'callback': '0'},
    'GTX 1080': {'rang': 128, 'callback': '1'},
    'GTX 980': {'rang': 178, 'callback': '2'},
    'GTX 1070': {'rang': 180, 'callback': '4'},
    'Radeon RX580': {'rang': 223, 'callback': '5'},
    'GTX 1060': {'rang': 239, 'callback': '6'},
    'GTX 980M': {'rang': 272, 'callback': '7'},
    'GTX 960': {'rang': 314, 'callback': '8'},
    'GTX 1050 Ti': {'rang': 320, 'callback': '9'},
    'GTX 970M': {'rang': 326, 'callback': '10'},
    'Quadro 3000M': {'rang': 333, 'callback': '11'},
    'GTX 940MX': {'rang': 352, 'callback': '12'},
    'GTX 1050': {'rang': 377, 'callback': '13'},
    'Radeon RX 560': {'rang': 382, 'callback': '14'},
    'Quadro M2200': {'rang': 387, 'callback': '15'},
    'GTX 965M': {'rang': 421, 'callback': '16'},
    'Radeon Pro 560': {'rang': 446, 'callback': '17'},
    'Quadro M2000M': {'rang': 448, 'callback': '18'},
    'GTX 940M': {'rang': 452, 'callback': '19'},
    'GTX 960M': {'rang': 453, 'callback': '20'}
}

Way = {
    "start": False,
    "begin_gpu": False,
    "gpu": False,
    "begin_memory": False,
    "memory": False,
    "begin_ram": False,
    "ram": False,
    "begin_cpu": False,
    "cpu": False,
    "begin_scr": False,
    "scr": False
}
ID = None
params = {
    "ScreenResolution": [0],
    "Cpu": [0],
    "Ram": [0],
    "Memory": [0],
    "Gpu": [0]}

def toPixelCount(a):
  t = a.find("x")
  n1 = ""
  for i in range(t):
    if a[t - 1 - i].isdigit():
      n1 = a[t - 1 - i] + n1
    else:
      break
  n2 = ""
  for i in range(t + 1, len(a)):
    if a[i].isdigit():
      n2 = n2 + a[i]
    else:
      break
  return int(n1)*int(n2)

def toFrequencyCount(a):
  t = a.rfind(" ")
  n = a[t + 1: -3]
  return float(n)

def toRamCount(a):
  return int(a[:-2])

def toMemoryCount(a):
  t = a.find("GB")
  n1 = '0'
  if t != -1:
   n1 = ''
   for i in range(t):
     if a[t - i - 1].isdigit():
       n1 = a[t - i - 1] + n1
     else:
       break
   n2 = '0'
  t = a.find("TB")
  if t != -1:
   n2 = ''
   for i in range(t):
    if a[t - i - 1].isdigit() or a[t - i - 1] == '.':
      n2 = a[t - i - 1] + n2
    else:
      break
  return int(n1) + int(n2[0]) * 1024

def toRangGPU(a):
   indx = gpu[gpu['Name'] == a].index.to_list()[0]
   return int(gpu.iloc[indx]['Rang'])


gpu = pd.read_csv("gpu.csv", sep=',')
gpu.columns = ['Name', 'Rang']
data = pd.read_csv("laptop_price.csv", sep=',')
data["ScreenResolution"] = data["ScreenResolution"].apply(toPixelCount)
data["Cpu"] = data["Cpu"].apply(toFrequencyCount)
data["Ram"] = data["Ram"].apply(toRamCount)
data["Memory"] = data["Memory"].apply(toMemoryCount)
data["Gpu"] = data["Gpu"].apply(toRangGPU)
data = data[[data.columns[4], data.columns[5], data.columns[6], data.columns[7], data.columns[8], data.columns[9], data.columns[12]]]
data.columns = ['Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory','Gpu', 'Price']
prices = data['Price']
features = data.drop(['Price', 'Inches'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
regr = LinearRegression()
regr.fit(x_train, y_train)

bot = telebot.TeleBot("6921207533:AAGTkbiWYJvMZ-X88746YSJFpBgSKa8X330")

@bot.message_handler(commands=['start'])
def send_start(message):
    global ID
    global Way
    Way = {
        "start": True,
        "begin_gpu": False,
        "gpu": False,
        "begin_memory": False,
        "memory": False,
        "begin_ram": False,
        "ram": False,
        "begin_cpu": False,
        "cpu": False,
        "begin_scr": False,
        "scr": False,
        }
    ID = message.chat.id
    btn = types.InlineKeyboardMarkup(row_width=2)
    btn.add(types.InlineKeyboardButton("Начать",callback_data='begin'))
    bot.send_message(ID, 'Здраствуйте,\nЭто бот Laptop Price Bot\nЯ могу предсказать стоимость ноутбука по выбранным характеристикам', reply_markup=btn)



@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    global params
    global ID
    global Way
    if call.data == 'begin' and Way['start'] and not Way['begin_gpu']:
        Way['begin_gpu'] = True
        params = {
            "ScreenResolution": [0],
            "Cpu": [0],
            "Ram": [0],
            "Memory": [0],
            "Gpu": [0]}
        main_kb = types.InlineKeyboardMarkup(row_width=2)

        buttons = []
        for name, value in gpus.items():
            buttons.append(types.InlineKeyboardButton(name, callback_data=value['callback']))

        main_kb.add(*buttons)

        bot.send_message(ID, 'Выберите видеокарту:', reply_markup=main_kb)
    if Way['begin_gpu'] and not Way['gpu']:
        for name, value in gpus.items():
            if value['callback'] in call.data:
                bot.send_message(call.message.chat.id, 'Видеокарта: {}'.format(name))
                params["Gpu"][0] = value['rang']
                Way['gpu'] = True
                break

    if Way['gpu'] and not Way['begin_memory']:
        Way['begin_memory'] = True
        main_kb = types.InlineKeyboardMarkup(row_width=2)
        buttons = []
        for name, value in memorys.items():
            buttons.append(types.InlineKeyboardButton(name, callback_data=value['callback']))

        main_kb.add(*buttons)

        bot.send_message(ID, 'Выберите количество памяти:', reply_markup=main_kb)

    if Way['begin_memory'] and not Way['memory']:
        for name, value in memorys.items():
            if value['callback'] in call.data:
                bot.send_message(call.message.chat.id, 'Память: {}'.format(name))
                params["Memory"][0] = value['count']
                Way['memory'] = True
                break

    if Way['memory'] and not Way['begin_ram']:
        Way['begin_ram'] = True
        main_kb = types.InlineKeyboardMarkup(row_width=2)
        buttons = []
        for name, value in rams.items():
            buttons.append(types.InlineKeyboardButton(name, callback_data=value['callback']))

        main_kb.add(*buttons)

        bot.send_message(ID, 'Выберите количество оперативной памяти:', reply_markup=main_kb)

    if Way['begin_ram'] and not Way['ram']:
        for name, value in rams.items():
            if value['callback'] in call.data:
                bot.send_message(call.message.chat.id, 'Оперативная память: {}'.format(name))
                params["Ram"][0] = value['count']
                Way['ram'] = True
                break

    if Way['ram'] and not Way['begin_cpu']:
        Way['begin_cpu'] = True
        main_kb = types.InlineKeyboardMarkup(row_width=2)
        buttons = []
        for name, value in cpus.items():
            buttons.append(types.InlineKeyboardButton(name, callback_data=value['callback']))

        main_kb.add(*buttons)

        bot.send_message(ID, 'Выберите процессор:', reply_markup=main_kb)

    if Way['begin_cpu'] and not Way['cpu']:
        for name, value in cpus.items():
            if value['callback'] in call.data:
                bot.send_message(call.message.chat.id, 'Процессор: {}'.format(name))
                params["Cpu"][0] = value['count']
                Way['cpu'] = True
                break

    if Way['cpu'] and not Way['begin_scr']:
        Way['begin_scr'] = True
        main_kb = types.InlineKeyboardMarkup(row_width=2)
        buttons = []
        for name, value in scrs.items():
            buttons.append(types.InlineKeyboardButton(name, callback_data=value['callback']))

        main_kb.add(*buttons)

        bot.send_message(ID, 'Выберите разрешение экрана:', reply_markup=main_kb)

    if Way['begin_scr'] and not Way['scr']:
        for name, value in scrs.items():
            if value['callback'] in call.data:
                bot.send_message(call.message.chat.id, 'Разрешение экрана: {}'.format(name))
                params["ScreenResolution"][0] = value['count']
                Way['scr'] = True
                break

    if Way['scr']:
        pred = regr.predict(pd.DataFrame(params))
        bot.send_message(call.message.chat.id, 'Стоимость такого ноутбука: \n{} рублей'.format(40 * round(regr.predict(pd.DataFrame(params))[0])))

bot.polling(none_stop=True)