from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5   # загружаем фреймворк TerraYolo
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# --- 0) ENV / TOKEN / YOLO ----------------------------------------------------
load_dotenv()
TOKEN = os.environ.get("TOKEN")  # ВАЖНО !!!!!
WORK_DIR = 'D:/UII/DataScience/16_OD/OD'
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)

# --- 0.1) Пресеты классов COCO для кнопок ------------------------------------
# ВАЖНО: значение должно быть строкой индексов (без скобок) или None (все классы)
CLASS_PRESETS = {
    # 0 - person
    "person":   ("Люди",        "0"),
    # 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
    "vehicles": ("Транспорт",   "1 2 3 5 7"),
    # 14 bird, 15 cat, 16 dog, 17 horse, 18 sheep, 19 cow, 20 elephant, 21 bear, 22 zebra, 23 giraffe
    "animals":  ("Животные",    "14 15 16 17 18 19 20 21 22 23"),
    # 9 traffic light, 11 stop sign
    "traffic":  ("Светофоры/знаки", "9 11"),
    # None -> без фильтра по классам (все классы)
    "all":      ("Все классы (сброс)", None),
}

def build_classes_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Люди", callback_data="cls:person"),
            InlineKeyboardButton("Транспорт", callback_data="cls:vehicles"),
        ],
        [
            InlineKeyboardButton("Животные", callback_data="cls:animals"),
            InlineKeyboardButton("Светофоры/знаки", callback_data="cls:traffic"),
        ],
        [
            InlineKeyboardButton("Сброс (все классы)", callback_data="cls:all"),
        ],
    ])

# --- 1) /start ---------------------------------------------------------------
async def start(update, context):
    user = update.effective_user  # получаем объект пользователя

    # Приветствие (с именем или username)
    welcome_text = f"👋 Привет, {user.first_name or user.username}!\n"
    welcome_text += "Добро пожаловать в KhiminArtemAI Bot 🚀\n\n"

    # Дальше идёт основная подсказка
    main_text = (
        "Пришлите фото для распознавания объектов.\n"
        "Чтобы выбрать тип объектов для фильтра, нажмите /objects"
    )

    await update.message.reply_text(welcome_text + main_text)


# --- 1.1) Меню выбора классов ------------------------------------------------
async def objects(update, context):
    await update.message.reply_text(
        "Выберите, какие объекты распознавать в режиме «по классам»:",
        reply_markup=build_classes_keyboard()
    )

async def on_cls(update, context):
    query = update.callback_query
    await query.answer()
    data = query.data  # например: "cls:person"
    key = data.split(":", 1)[1]

    if key not in CLASS_PRESETS:
        await query.edit_message_text("Неизвестный пресет.")
        return

    preset_name, class_str = CLASS_PRESETS[key]
    # Сохраняем выбор пользователя
    context.user_data['selected_classes_key']  = key         # 'person' | 'vehicles' | 'animals' | 'traffic' | 'all'
    context.user_data['selected_classes_name'] = preset_name  # текст для пользователя
    context.user_data['selected_classes_str']  = class_str    # '0 2 7' | None

    if class_str is None:
        await query.edit_message_text("Фильтр классов: сброшен (все классы).")
    else:
        await query.edit_message_text(f"Фильтр классов: {preset_name} (COCO: {class_str}).")

# --- 2) /help и ответ на текст ----------------------------------------------
async def help(update, context):
    help_text = (
        "🆘 *Помощь по командам YOLO Bot*\n\n"
        "/start – Приветствие и начало работы\n"
        "/objects – Выбор типов объектов для фильтрации (люди, транспорт, животные, светофоры)\n"
        "/help – Список доступных команд и описание системы\n\n"
        "📸 Просто отправьте фото или изображение (как фото или как документ), "
        "и бот выполнит распознавание объектов.\n\n"
        "⚙️ Система использует модель *YOLOv5x* и позволяет варьировать:\n"
        " - Уровень достоверности (conf-threshold)\n"
        " - Порог IoU для NMS (iou-threshold)\n"
        " - Классы объектов (по выбору пользователя)\n\n"
        "🔒 Личные данные (токен, окружение) не публикуются. "
        "Результаты детекции сохраняются локально в `runs/detect/`."
    )

    await update.message.reply_text(help_text, parse_mode="Markdown")


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
    tg_file = None
    image_name = None

    # Сжатое фото
    if update.message and update.message.photo:
        tg_file = await update.message.photo[-1].get_file()
        image_name = os.path.basename(tg_file.file_path) if tg_file.file_path else 'photo.jpg'

    # Несжатое изображение как документ (оригинал)
    elif update.message and update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith('image/'):
        tg_file = await update.message.document.get_file()
        image_name = update.message.document.file_name or (os.path.basename(tg_file.file_path) if tg_file.file_path else 'document_image.jpg')

    else:
        await context.bot.deleteMessage(message_id=my_message.message_id, chat_id=update.message.chat_id)
        await update.message.reply_text('Похоже, это не изображение. Пришлите, пожалуйста, фото или картинку.')
        return

    os.makedirs('images', exist_ok=True)
    image_path = os.path.join('images', image_name)
    await tg_file.download_to_drive(image_path)
    image_name = os.path.basename(image_path)

    # === ШАГ C. Базовый словарь параметров для YOLO ==========================
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

        out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        yolov5.run(test_dict, exp_type='test')

        out_path = f"{WORK_DIR}/yolov5/runs/detect/{run_name}/{image_name}"
        if os.path.exists(out_path):
            result_paths_conf.append((c, out_path))

    # === ШАГ D2. Прогоны по IoU (три уровня) ==================================
    iou_list = [0.01, 0.5, 0.99]
    result_paths_iou = []

    for i in iou_list:
        test_dict['conf-thres'] = 0.25      # фиксируем уверенность для честного сравнения NMS
        test_dict['iou-thres']  = i
        run_name = f"iou_{int(i*100):03d}"
        test_dict['name'] = run_name

        out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        yolov5.run(test_dict, exp_type='test')

        out_path = f"{WORK_DIR}/yolov5/runs/detect/{run_name}/{image_name}"
        if os.path.exists(out_path):
            result_paths_iou.append((i, out_path))

    # === ШАГ D3. Прогон по классам (с учётом выбранного пресета) ==============
    # Читаем выбор пользователя; по умолчанию — "Люди"
    selected_key  = context.user_data.get('selected_classes_key', 'person')
    selected_name = context.user_data.get('selected_classes_name', CLASS_PRESETS['person'][0])
    selected_str  = context.user_data.get('selected_classes_str', CLASS_PRESETS['person'][1])

    # Настройки для "классового" прогона
    test_dict['conf-thres'] = 0.5
    test_dict['iou-thres']  = 0.45
    run_name_classes = f"classes_{selected_key}"
    test_dict['name'] = run_name_classes

    # ВАЖНО: test_dict['classes'] только если есть фильтр (не None)
    if selected_str is not None:
        test_dict['classes'] = selected_str  # строка, например "0 2 7"
    elif 'classes' in test_dict:
        del test_dict['classes']

    out_dir = f"{WORK_DIR}/yolov5/runs/detect/{run_name_classes}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    yolov5.run(test_dict, exp_type='test')
    out_path_classes = f"{WORK_DIR}/yolov5/runs/detect/{run_name_classes}/{image_name}"

    # После запуска убираем 'classes', чтобы не повлияло на будущие проги
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
    await update.message.reply_text(f"2. Распознавание с различным уровнем метрики “пересечение-на-объединение” iou = {iou_list}")
    for i, p in result_paths_iou:
        await update.message.reply_text(f"IoU = {i}")
        await update.message.reply_photo(p)

    # 3) Классы (с учётом выбора через /objects)
    if selected_str is None:
        await update.message.reply_text(
            "3. Распознавание по классам: все классы (фильтр отключён)."
        )
    else:
        await update.message.reply_text(
            f"3. Распознавание по выбранным классам: {selected_name} (COCO: {selected_str})."
        )
    if os.path.exists(out_path_classes):
        await update.message.reply_photo(out_path_classes)
    else:
        await update.message.reply_text("Не удалось сформировать результат для режима «по классам».")


# --- 4) Точка входа ----------------------------------------------------------
def main():
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("objects", objects))

    # Инлайн-кнопки (callback)
    application.add_handler(CallbackQueryHandler(on_cls, pattern=r"^cls:"))

    # Изображения: несжатые (документ image/*) + сжатые (photo)
    application.add_handler(MessageHandler(filters.Document.IMAGE, detection, block=False))
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))

    # Текст
    application.add_handler(MessageHandler(filters.TEXT, help))

    application.run_polling()  # остановка CTRL + C


if __name__ == "__main__":
    main()
