from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5   # загружаем фреймворк TerraYolo

# --- 0) ENV / TOKEN / YOLO ----------------------------------------------------
load_dotenv()
TOKEN = os.environ.get("TOKEN")  # ВАЖНО !!!!!
WORK_DIR = 'D:/UII/DataScience/16_OD/OD'
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)


# --- 1) /start ---------------------------------------------------------------
async def start(update, context):
    await update.message.reply_text('Пришлите фото для распознавания объектов')


# --- 2) /help и ответ на текст ----------------------------------------------
async def help(update, context):
    # Отвечаем текстом пользователя (без изменения названий ваших функций/переменных)
    if update and update.message and update.message.text:
        await update.message.reply_text(update.message.text)
    else:
        await update.message.reply_text("Отправьте фото или изображение (как фото или как документ).")


# --- 3) Обработка изображения ------------------------------------------------
async def detection(update, context):
    # === ШАГ A. Подготовка окружения (чистим прошлые результаты) =============
    try:
        shutil.rmtree('images')
    except Exception:
        pass
    try:
        shutil.rmtree(f'{WORK_DIR}/yolov5/runs')
    except Exception:
        pass

    # Сообщение о начале обработки
    my_message = await update.message.reply_text(
        'Мы получили от тебя изображение. Идет распознавание объектов...'
    )

    # === ШАГ B. Скачиваем изображение из Telegram ============================
    # Поддерживаем два варианта: сжатое фото (filters.PHOTO) и документ-изображение (filters.Document.IMAGE).
    tg_file = None
    image_name = None

    if update.message and update.message.photo:
        # Сжатое фото: берем максимальный размер
        tg_file = await update.message.photo[-1].get_file()
        # Имя файла по пути телеграма (если нет — дадим дефолт)
        image_name = os.path.basename(tg_file.file_path) if tg_file.file_path else 'photo.jpg'

    elif update.message and update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith('image/'):
        # Несжатое изображение как документ
        tg_file = await update.message.document.get_file()
        # Для документов Telegram часто присылает оригинальное имя
        image_name = update.message.document.file_name or (os.path.basename(tg_file.file_path) if tg_file.file_path else 'document_image.jpg')

    else:
        # Если это не изображение — сообщим и выйдем
        await context.bot.deleteMessage(message_id=my_message.message_id, chat_id=update.message.chat_id)
        await update.message.reply_text('Похоже, это не изображение. Пришлите, пожалуйста, фото или картинку.')
        return

    os.makedirs('images', exist_ok=True)
    image_path = os.path.join('images', image_name)
    await tg_file.download_to_drive(image_path)  # сохраняем файл в ./images
    # Имя, под которым YOLO сохранит размеченную картинку:
    # как правило, совпадает с именем входного файла
    image_name = os.path.basename(image_path)

    # === ШАГ C. Базовый словарь параметров для YOLO ==========================
    # ВНИМАНИЕ: в YOLOv5 правильные ключи CLI — в стиле kebab-case:
    #   conf-thres, iou-thres, name, classes, source, weights, ...
    test_dict = dict()
    test_dict['weights'] = 'yolov5x.pt'   # сильные веса
    test_dict['source']  = 'images'       # папка с входными изображениями

    # === ШАГ D1. Прогоны по conf (три уровня) =================================
    conf_list = [0.01, 0.5, 0.99]
    result_paths_conf = []

    for c in conf_list:
        test_dict['conf-thres'] = c
        run_name = f"conf_{int(c*100):03d}"
        test_dict['name'] = run_name

        # Чистим целевую папку прогона (аналог --exist-ok)
        out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        # Запуск детекции
        yolov5.run(test_dict, exp_type='test')

        # Путь к результату
        out_path = f"{WORK_DIR}/yolov5/runs/detect/{run_name}/{image_name}"
        if os.path.exists(out_path):
            result_paths_conf.append((c, out_path))

    # === ШАГ D2. Прогоны по IoU (три уровня) ==================================
    iou_list = [0.01, 0.5, 0.99]
    result_paths_iou = []

    for i in iou_list:
        test_dict['iou-thres'] = i
        # Можно оставить последний conf-thres или задать фиксированный тут, по желанию:
        test_dict['conf-thres'] = 0.25  #  уверенность для iou-прогонов
        run_name = f"iou_{int(i*100):03d}"
        test_dict['name'] = run_name

        out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        yolov5.run(test_dict, exp_type='test')

        out_path = f"{WORK_DIR}/yolov5/runs/detect/{run_name}/{image_name}"
        if os.path.exists(out_path):
            result_paths_iou.append((i, out_path))

    # === ШАГ D3. Прогон по классам (только люди) ==============================
    # COCO: 0 — person
    test_dict['conf-thres'] = 0.6          # можно чуть поднять уверенность
    test_dict['iou-thres']  = 0.45         # вернём дефолт NMS, чтобы не плодить дубли
    test_dict['classes']    = '0'          # ВАЖНО: строка без скобок! (а не [0])

    run_name_classes = "classes_person"
    test_dict['name'] = run_name_classes

    out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name_classes}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    yolov5.run(test_dict, exp_type='test')
    out_path_classes = f"{WORK_DIR}/yolov5/runs/detect/{run_name_classes}/{image_name}"

    # чтобы параметр 'classes' не влиял на другие запуски потом:
    if 'classes' in test_dict:
        del test_dict['classes']

    # === ШАГ E. Убираем "идет распознавание..." ===============================
    await context.bot.deleteMessage(message_id=my_message.message_id, chat_id=update.message.chat_id)

    # === ШАГ F. Вывод результатов пользователю ================================
    # 1) conf
    await update.message.reply_text(f"1. Распознавание с различным уровнем достоверности conf = {conf_list}")
    for c, p in result_paths_conf:
        await update.message.reply_text(f"conf = {c}")
        await update.message.reply_photo(p)

    # 2) IoU
    await update.message.reply_text(f"2. Распознавание с различным уровнем метрики “пересечения на объединение” iou = {iou_list}")
    for i, p in result_paths_iou:
        await update.message.reply_text(f"IoU = {i}")
        await update.message.reply_photo(p)

    # 3) Классы (только люди)
    await update.message.reply_text(
        "3. Распознавание определенных предобученных классов.\n"
        "Пример: только люди (COCO class=0). Параметр передаём через словарь: test_dict['classes'] = [0]"
    )
    if os.path.exists(out_path_classes):
        await update.message.reply_photo(out_path_classes)
    else:
        await update.message.reply_text("Не удалось сформировать результат для детекции только по классу 'person'.")


# --- 4) Точка входа ----------------------------------------------------------
def main():
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # Команды
    application.add_handler(CommandHandler("start", start))

    # ВАЖНО: добавляем обработчик для несжатых изображений-документов (image/*)
    application.add_handler(MessageHandler(filters.Document.IMAGE, detection, block=False))
    # И для сжатых фото
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))

    # Текст
    application.add_handler(MessageHandler(filters.TEXT, help))

    application.run_polling()  # остановка CTRL + C

# начало
if __name__ == "__main__":
    main()
