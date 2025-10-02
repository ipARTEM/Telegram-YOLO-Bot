from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5             # загружаем фреймворк TerraYolo

# возьмем переменные окружения из .env
load_dotenv()

# загружаем токен бота
TOKEN =  os.environ.get("TOKEN") # ВАЖНО !!!!!

# инициализируем класс YOLO
WORK_DIR = 'D:/UII/DataScience/16_OD/OD'
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)


# функция команды /start
async def start(update, context):
    await update.message.reply_text('Пришлите фото для распознавания объектов')

# функция для работы с текстом
async def help(update, context):
    await update.message.reply_text(update)
    await update.message.reply_text(update.message.text)    # Отправить текст



# функция обработки изображения
async def detection(update, context):
    # удаляем папку images с предыдущим загруженным изображением и папку runs с результатом предыдущего распознавания
    try:
        shutil.rmtree('images') 
        shutil.rmtree(f'{WORK_DIR}/yolov5/runs') 
    except:
        pass

    my_message = await update.message.reply_text('Мы получили от тебя фотографию. Идет распознавание объектов...')
    # получение файла из сообщения
    new_file = await update.message.photo[-1].get_file()

    # имя файла на сервере
    os.makedirs('images', exist_ok=True)
    image_name = str(new_file['file_path']).split("/")[-1]
    image_path = os.path.join('images', image_name)
    # скачиваем файл с сервера Telegram в папку images
    await new_file.download_to_drive(image_path)

    # создаем словарь с параметрами
    test_dict = dict()
    test_dict['weights'] = 'yolov5x.pt'     # Самые сильные веса yolov5x.pt, вы также можете загрузить версии: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt (в порядке возрастания)

    test_dict['source'] = 'images'          # папка, в которую загружаются присланные в бота изображения
    # test_dict['conf'] = 0.85              # порог распознавания
    # test_dict['classes'] = '50 39'        # классы, которые будут распознаны


    # вызов функции detect из класса TerraYolo)
    yolov5.run(test_dict, exp_type='test') 

    # удаляем предыдущее сообщение от бота
    await context.bot.deleteMessage(message_id = my_message.message_id, # если не указать message_id, то удаляется последнее сообщение
                                    chat_id = update.message.chat_id) # если не указать chat_id, то удаляется последнее сообщение

    # отправляем пользователю результат
    await update.message.reply_text('Распознавание объектов завершено') # отправляем пользователю результат 
    await update.message.reply_photo(f"{WORK_DIR}/yolov5/runs/detect/exp/{image_name}") # отправляем пользователю результат изображение



def main():

    # точка входа в приложение
    application = Application.builder().token(TOKEN).build() # создаем объект класса Application
    print('Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))
    # добавляем обработчик изображений, которые загружаются в Telegram в СЖАТОМ формате
    # (выбирается при попытке прикрепления изображения к сообщению)
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.add_handler(MessageHandler(filters.TEXT, help))

    application.run_polling() # запускаем бота (остановка CTRL + C)


if __name__ == "__main__":
    main()
