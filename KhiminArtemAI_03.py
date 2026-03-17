from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import os
import io
import shutil
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple
# from TerraYolo.TerraYolo import TerraYoloV5   # фреймворк TerraYolo

import sys  # импортируем sys для добавления локальной папки yolov5 в пути Python
from pathlib import Path  # импортируем Path для удобной работы с путями

from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats, BotCommandScopeDefault


import logging  # логирование для стабильной диагностики
from logging.handlers import RotatingFileHandler  # ротация логов, чтобы файл не рос бесконечно
from telegram.error import NetworkError, TimedOut, RetryAfter  # типовые ошибки сети Telegram


# === 0) ENV / TOKEN / YOLO ===================================================
load_dotenv()
TOKEN = os.environ.get("TOKEN")  # ВАЖНО !!!!!  токен бота


# --- 0.1) LOGGING -------------------------------------------------------------  # раздел логирования
logger = logging.getLogger(__name__)  # создаём логгер текущего модуля

logging.basicConfig(  # базовая настройка логирования
    level=logging.INFO,  # уровень логирования INFO
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",  # единый формат логов
)  # конец basicConfig

file_handler = RotatingFileHandler(  # файл-обработчик с ротацией
    filename="model.log",  # пишем логи в model.log (как у тебя уже есть файл)
    maxBytes=2 * 1024 * 1024,  # максимум 2 МБ на файл лога
    backupCount=5,  # хранить до 5 резервных файлов
    encoding="utf-8",  # кодировка UTF-8
)  # конец RotatingFileHandler

file_handler.setFormatter(  # задаём форматирование логов для файла
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")  # формат логов
)  # конец setFormatter

logging.getLogger().addHandler(file_handler)  # добавляем запись логов в файл для root-логгера

logging.getLogger("httpx").setLevel(logging.WARNING)  # отключаем информационный шум httpx
logging.getLogger("httpcore").setLevel(logging.WARNING)  # отключаем информационный шум httpcore

# --- 0.2) Подключение локальной папки yolov5 ---------------------------------  # раздел подключения локального YOLOv5
CURRENT_DIR = Path(__file__).resolve().parent  # получаем абсолютный путь к папке текущего файла
YOLOV5_DIR = CURRENT_DIR / "yolov5"  # формируем путь к локальной папке yolov5

if str(YOLOV5_DIR) not in sys.path:  # если путь к yolov5 ещё не добавлен в sys.path
    sys.path.insert(0, str(YOLOV5_DIR))  # добавляем путь в начало sys.path для корректного импорта модулей yolov5

# from detect import run as yolov5_detect_run  # импортируем функцию run из локального yolov5/detect.py


class TerraYoloV5:
    """Локальный адаптер вместо внешнего TerraYoloV5 для совместимости с текущей логикой проекта."""  # описание класса

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir  # сохраняем рабочую директорию проекта

    def run(self, test_dict: dict, mode: str = "test") -> None:
        """Запускает локальный yolov5/detect.py через совместимый интерфейс test_dict."""  # описание метода
        weights = test_dict.get("weights", "yolov5x.pt")  # получаем путь к весам модели
        source = test_dict.get("source")  # получаем путь к источнику изображений
        name = test_dict.get("name", "exp")  # получаем имя папки результата
        conf_thres = float(test_dict.get("conf-thres", 0.25))  # получаем confidence threshold
        iou_thres = float(test_dict.get("iou-thres", 0.45))  # получаем IoU threshold
        classes_raw = test_dict.get("classes")  # получаем строку или список классов

        if classes_raw is None:  # если классы не указаны
            classes = None  # передаём None, чтобы детектить все классы
        elif isinstance(classes_raw, str):  # если классы пришли строкой вида "0 1 2"
            classes = [int(item) for item in classes_raw.split() if item.strip()]  # преобразуем строку в список целых индексов
        elif isinstance(classes_raw, (list, tuple, set)):  # если классы уже в виде коллекции
            classes = [int(item) for item in classes_raw]  # приводим все значения к int
        else:  # если тип неожиданный
            classes = None  # безопасно сбрасываем фильтр классов

        weights_path = Path(self.work_dir) / weights  # формируем абсолютный путь к весам модели внутри проекта
        project_path = Path(self.work_dir) / "yolov5" / "runs" / "detect"  # формируем путь к папке результатов detect

        logger.info(  # пишем в лог факт запуска YOLOv5
            "Запуск локального yolov5: weights=%s, source=%s, name=%s, conf=%.3f, iou=%.3f, classes=%s",
            weights_path,
            source,
            name,
            conf_thres,
            iou_thres,
            classes,
        )  # конец logger.info

        yolov5_detect_run(  # запускаем локальный детект напрямую через функцию run из detect.py
            weights=str(weights_path),  # передаём путь к весам модели
            source=str(source),  # передаём путь к папке с изображениями
            data=str(YOLOV5_DIR / "data" / "coco128.yaml"),  # передаём yaml с описанием датасета COCO
            imgsz=(640, 640),  # задаём размер изображения для инференса
            conf_thres=conf_thres,  # передаём confidence threshold
            iou_thres=iou_thres,  # передаём IoU threshold
            max_det=1000,  # оставляем лимит на количество детекций
            device="",  # используем доступное устройство автоматически
            view_img=False,  # не показываем окно OpenCV
            save_txt=False,  # не сохраняем txt-разметку
            save_csv=False,  # не сохраняем csv
            save_conf=False,  # не сохраняем confidence в txt
            save_crop=False,  # не сохраняем кропы
            nosave=False,  # сохраняем результирующие изображения
            classes=classes,  # передаём фильтр классов
            agnostic_nms=False,  # не включаем class-agnostic NMS
            augment=False,  # не включаем augmented inference
            visualize=False,  # не сохраняем feature maps
            update=False,  # не обновляем веса
            project=str(project_path),  # указываем папку, куда сохранять результаты
            name=str(name),  # указываем имя конкретной подпапки запуска
            exist_ok=True,  # разрешаем использовать уже существующую папку после её очистки снаружи
            line_thickness=3,  # толщина рамки вокруг объекта
            hide_labels=False,  # показываем названия классов
            hide_conf=False,  # показываем confidence
            half=False,  # не включаем fp16 по умолчанию
            dnn=False,  # не используем OpenCV DNN backend
            vid_stride=1,  # для изображений не влияет, оставляем 1
        )  # конец вызова yolov5_detect_run



WORK_DIR = 'D:/UII/DataScience/16_OD/OD'
os.makedirs(WORK_DIR, exist_ok=True)

yolov5 = TerraYoloV5(work_dir=WORK_DIR)  # фреймворк и рабочая папка

# --- Администраторы (замени на свои Telegram id) -----------------------------
ADMIN_USER_IDS = {
    # пример: 123456789
}

# --- Настройки «долгих» задач ------------------------------------------------
DETECT_TIMEOUT_SEC = 180         # тайм-аут на одну задачу YOLO
CACHE_MAX_ITEMS = 200            # максимум ключей в кеш-индексе
CACHE_DIR = os.path.join(WORK_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Пресеты классов COCO для кнопок -----------------------------------------
# Значение: str индексов ("0 1 2") или None (все классы)
CLASS_PRESETS = {
    "person":   ("Люди",              "0"),
    "vehicles": ("Транспорт",         "1 2 3 5 7"),
    "animals":  ("Животные",          "14 15 16 17 18 19 20 21 22 23"),
    "traffic":  ("Светофоры/знаки",   "9 11"),
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

# === 0.5) Вспомогательные функции ===========================================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USER_IDS

def safe_basename(path: str) -> str:
    return os.path.basename(path).replace("\\", "/").split("/")[-1]

# === 0.6) Семафоры пользователей: один запрос за раз =========================
_user_locks: Dict[int, asyncio.Lock] = {}

def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

# === 0.7) Сервис детекции с кешем (ООП) ======================================
class DetectionService:
    """
    Класс берёт на себя:
      - формирование параметров YOLO
      - кеширование результатов (по хешу изображения + параметров)
      - запуск TerraYoloV5 и возврат путей к результатам
    """
    def __init__(self,
                 work_dir: str,
                 cache_dir: str,
                 yolov5_obj: TerraYoloV5,
                 cache_max_items: int = 200):
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.yolov5 = yolov5_obj
        self.cache_index: List[str] = []  # LRU-список ключей
        self.cache_max_items = cache_max_items

    # ---------- Хеш ключа кеша ----------
    def _calc_cache_key(self,
                        image_bytes: bytes,
                        mode: str,
                        weights: str,
                        conf: float,
                        iou: float,
                        classes_str: Optional[str]) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        payload = f"|mode={mode}|weights={weights}|conf={conf}|iou={iou}|classes={classes_str or 'ALL'}|"
        h.update(payload.encode('utf-8'))
        return h.hexdigest()

    # ---------- Проверка кеша ----------
    def _cache_paths(self, cache_key: str) -> List[str]:
        folder = os.path.join(self.cache_dir, cache_key)
        if not os.path.isdir(folder):
            return []
        # возвращаем все картинки из папки кеша
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    # ---------- Сохранить в кеш ----------
    def _save_to_cache(self, cache_key: str, result_paths: List[str]) -> List[str]:
        folder = os.path.join(self.cache_dir, cache_key)
        os.makedirs(folder, exist_ok=True)
        out_paths = []
        for i, src in enumerate(result_paths, 1):
            if not os.path.isfile(src):
                continue
            dst = os.path.join(folder, f"result_{i}_{safe_basename(src)}")
            try:
                shutil.copy2(src, dst)
                out_paths.append(dst)
            except Exception:
                pass
        # обновляем LRU
        if cache_key in self.cache_index:
            self.cache_index.remove(cache_key)
        self.cache_index.insert(0, cache_key)
        # чистим старое
        while len(self.cache_index) > self.cache_max_items:
            old_key = self.cache_index.pop()
            shutil.rmtree(os.path.join(self.cache_dir, old_key), ignore_errors=True)
        return out_paths

    # ---------- Запуск YOLO с параметрами ----------
    async def run_detection(self,
                            image_path: str,
                            image_bytes: bytes,
                            mode: str,  # "fast" | "pro"
                            selected_classes_str: Optional[str]) -> List[str]:
        """
        Возвращает список путей к изображениям-результатам для отправки пользователю.
        """
        weights = 'yolov5x.pt'

        # Параметры по умолчанию
        conf_base = 0.5
        iou_base  = 0.45

        # Ключ кеша для «быстрого» режима
        cache_key = self._calc_cache_key(
            image_bytes=image_bytes,
            mode=mode,
            weights=weights,
            conf=conf_base,
            iou=iou_base,
            classes_str=selected_classes_str
        )

        # 1) Пробуем кеш
        cached = self._cache_paths(cache_key)
        if cached:
            return cached

        # 2) Готовим общую входную папку для TerraYolo
        images_dir = os.path.join(self.work_dir, "images_in")
        shutil.rmtree(images_dir, ignore_errors=True)
        os.makedirs(images_dir, exist_ok=True)
        local_name = safe_basename(image_path)
        local_path = os.path.join(images_dir, local_name)
        shutil.copy2(image_path, local_path)

        # Базовый словарь параметров
        test_dict = {
            'weights': weights,
            'source': images_dir,
        }

        result_paths: List[str] = []

        async def _run_once(name: str,
                            conf: float,
                            iou: float,
                            classes: Optional[str]) -> Optional[str]:
            test_dict['name'] = name
            test_dict['conf-thres'] = conf
            test_dict['iou-thres'] = iou

            if classes is not None:
                test_dict['classes'] = classes
            elif 'classes' in test_dict:
                del test_dict['classes']

            # чистим выход
            out_dir = os.path.join(self.work_dir, "yolov5", "runs", "detect", name)
            shutil.rmtree(out_dir, ignore_errors=True)

            # Запускаем с тайм-аутом
            await asyncio.wait_for(asyncio.to_thread(self.yolov5.run, test_dict, 'test'),
                                   timeout=DETECT_TIMEOUT_SEC)

            out_img = os.path.join(out_dir, local_name)
            return out_img if os.path.exists(out_img) else None

        # 3) Режимы
        if mode == "fast":
            out = await _run_once("fast_single", conf_base, iou_base, selected_classes_str)
            if out:
                result_paths.append(out)

        elif mode == "pro":
            # a) Грид по conf
            for c in (0.01, 0.50, 0.99):
                out = await _run_once(f"conf_{int(c*100):03d}", c, iou_base, selected_classes_str)
                if out:
                    result_paths.append(out)
            # b) Грид по IoU (фиксируем conf)
            for i in (0.01, 0.50, 0.99):
                out = await _run_once(f"iou_{int(i*100):03d}", conf_base, i, selected_classes_str)
                if out:
                    result_paths.append(out)

        # 4) Сохраняем в кеш «под ключ fast» (в т.ч. для pro — кешируем весь пакет)
        if result_paths:
            return self._save_to_cache(cache_key, result_paths)
        return []

# Инициализируем сервис
detector = DetectionService(WORK_DIR, CACHE_DIR, yolov5, CACHE_MAX_ITEMS)

# === 1) /start ===============================================================
async def start(update, context):
    user = update.effective_user
    welcome_text = f"👋 Привет, {user.first_name or user.username}!\nДобро пожаловать в Object Detection Bot 🚀\n\n"
    main_text = (
        "Пришлите фото для распознавания объектов.\n"
        "Чтобы выбрать тип объектов для фильтра — нажмите /objects\n"
        "Справка по командам — /help"
    )
    await update.message.reply_text(welcome_text + main_text)

# === 1.1) Выбор классов ======================================================
async def objects(update, context):
    await update.message.reply_text(
        "Выберите, какие объекты распознавать в режиме «по классам»:",
        reply_markup=build_classes_keyboard()
    )

async def on_cls(update, context):
    query = update.callback_query
    await query.answer()
    key = query.data.split(":", 1)[1]
    if key not in CLASS_PRESETS:
        await query.edit_message_text("Неизвестный пресет.")
        return

    preset_name, class_str = CLASS_PRESETS[key]
    context.user_data['selected_classes_key']  = key
    context.user_data['selected_classes_name'] = preset_name
    context.user_data['selected_classes_str']  = class_str

    if class_str is None:
        await query.edit_message_text("Фильтр классов: сброшен (все классы).")
    else:
        await query.edit_message_text(f"Фильтр классов: {preset_name} (COCO: {class_str}).")

# === 2) /help =================================================================
async def help(update, context):
    help_text = (
        "🆘 *Помощь по командам KhiminArtem Bot*\n\n"
        "/start – Приветствие и начало работы\n"
        "/objects – Выбор типов объектов для фильтрации (люди, транспорт, животные, светофоры)\n"
        "/help – Список доступных команд и описание системы\n"
        "/mode – Показать ваш текущий режим (fast/pro)\n"
        "/fast – Включить быстрый режим (один прогон)\n"
        "/pro – Тяжёлый режим (грид по conf и IoU, *только админ*)\n\n"
        "📸 Отправьте фото (как фото или документ) — бот выполнит распознавание объектов.\n\n"
        "⚙️ Модель: *Распознавания*. Настройки:\n"
        " - conf-threshold (уверенность детекции)\n"
        " - IoU-threshold для NMS\n"
        " - Фильтрация по классам COCO\n\n"
        "💾 Встроен кеш результатов по хешу изображения и параметрам.\n"
        "🔒 Тяжёлые режимы доступны только администраторам."
        "\n"
        "✅🧑🏻‍💻 Больше информации вы можете найти на сайте ah8.ru.\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

# === 2.1) Режимы пользователя ===============================================
async def show_mode(update, context):
    # fast/pro хранится в user_data['mode']; по умолчанию fast
    mode = context.user_data.get('mode', 'fast')
    role = "admin" if is_admin(update.effective_user.id) else "user"
    await update.message.reply_text(f"Текущий режим: *{mode}* (роль: {role})", parse_mode="Markdown")

async def set_fast(update, context):
    context.user_data['mode'] = 'fast'
    await update.message.reply_text("Режим установлен: *fast* (быстрый однопроходный).", parse_mode="Markdown")

async def set_pro(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда доступна только администратору.")
        return
    context.user_data['mode'] = 'pro'
    await update.message.reply_text("Режим установлен: *pro* (тяжёлый, грид по conf/IoU).", parse_mode="Markdown")

# === 3) Обработка изображения (с кешем/лимитами) =============================
async def detection(update, context):
    user = update.effective_user
    lock = get_user_lock(user.id)

    if lock.locked():
        await update.message.reply_text("⏳ У вас уже выполняется задача. Дождитесь её завершения, пожалуйста.")
        return

    async with lock:
        # Получаем файл из Telegram
        tg_file = None
        image_name = None

        if update.message and update.message.photo:
            tg_file = await update.message.photo[-1].get_file()
            image_name = safe_basename(tg_file.file_path) if tg_file.file_path else 'photo.jpg'
        elif update.message and update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith('image/'):
            tg_file = await update.message.document.get_file()
            image_name = update.message.document.file_name or (safe_basename(tg_file.file_path) if tg_file.file_path else 'document_image.jpg')
        else:
            await update.message.reply_text('Похоже, это не изображение. Пришлите, пожалуйста, фото или картинку.')
            return

        # Скачиваем в локальную tmp-папку
        tmp_dir = os.path.join(WORK_DIR, "tmp_in")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        image_path = os.path.join(tmp_dir, image_name)
        await tg_file.download_to_drive(image_path)

        # Читаем байты для хеша
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Параметры пользователя
        selected_str = context.user_data.get('selected_classes_str', CLASS_PRESETS['person'][1])
        mode = context.user_data.get('mode', 'fast')
        if mode == 'pro' and not is_admin(user.id):
            mode = 'fast'  # безопасность: только админ может pro

        # Статусные сообщения
        processing_msg = await update.message.reply_text("📥 Изображение получено. Проверяю кеш…")

        try:
            # Запуск детекции через сервис
            results = await detector.run_detection(
                image_path=image_path,
                image_bytes=image_bytes,
                mode=mode,
                selected_classes_str=selected_str
            )
        except asyncio.TimeoutError:
            await processing_msg.edit_text("❌ Время обработки истекло. Попробуйте ещё раз (или используйте /fast).")
            return
        except Exception as e:
            await processing_msg.edit_text(f"❌ Ошибка обработки: {e}")
            return

        # Вывод результатов
        if not results:
            await processing_msg.edit_text("Готово. Объекты не найдены или результат не сформирован.")
            return

        await processing_msg.edit_text(f"Готово. Режим: *{mode}*. Отправляю результаты…", parse_mode="Markdown")
        for p in results:
            await update.message.reply_photo(p)




# === Регистрация подсказок команд (меню /) ====================================
async def _setup_commands(app):
    # Команды для ЛИЧНЫХ чатов (полный список)
    private_commands = [
        BotCommand("start",  "Приветствие и инструкция"),
        BotCommand("help",   "Помощь по командам и описание системы"),
        BotCommand("objects","Выбор типов объектов (классы COCO)"),
        BotCommand("mode",   "Показать текущий режим (fast/pro)"),
        BotCommand("fast",   "Включить быстрый режим"),
        BotCommand("pro",    "Включить тяжёлый режим (только админ)"),
    ]
    await app.bot.set_my_commands(private_commands, scope=BotCommandScopeAllPrivateChats())

    # Команды для ГРУПП (обычно короче)
    group_commands = [
        BotCommand("start",  "Приветствие"),
        BotCommand("help",   "Помощь по командам"),
        BotCommand("objects","Выбор классов распознавания"),
        BotCommand("mode",   "Показать режим"),
    ]
    await app.bot.set_my_commands(group_commands, scope=BotCommandScopeAllGroupChats())

    # Базовый дефолт (на всякий случай)
    await app.bot.set_my_commands(private_commands, scope=BotCommandScopeDefault())


async def error_handler(update, context) -> None:  # обработчик ошибок приложения
    """Обработчик ошибок Telegram-бота, чтобы сетевые сбои (502/таймаут) не выглядели как падение."""  # описание
    err: Exception = context.error  # берём исключение из контекста

    if isinstance(err, RetryAfter):  # если Telegram просит подождать (лимиты)
        logger.warning("Telegram RetryAfter: %s seconds", err.retry_after)  # пишем предупреждение в лог
        return  # выходим без traceback

    if isinstance(err, (NetworkError, TimedOut)):  # если сетевой сбой/таймаут/502
        logger.warning("Network error while polling: %s", err)  # логируем кратко без traceback
        return  # выходим без traceback

    logger.exception("Unhandled exception in bot: %s", err)  # все остальные ошибки пишем с traceback


def main():
    application = (
        Application.builder()
        .token(TOKEN)
        .post_init(_setup_commands)   # <-- ВАЖНО: здесь, на builder
        .build()
    )
    print('Бот запущен...')

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("objects", objects))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("mode", show_mode))
    application.add_handler(CommandHandler("fast", set_fast))
    application.add_handler(CommandHandler("pro", set_pro))

    # Инлайн-кнопки (callback)
    application.add_handler(CallbackQueryHandler(on_cls, pattern=r"^cls:"))

    # Медиа
    application.add_handler(MessageHandler(filters.Document.IMAGE, detection, block=False))
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))

    # Текст (лучше исключить команды, чтобы /help не ловился ещё и этим хендлером)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, help))

    application.add_error_handler(error_handler)  # регистрируем обработчик ошибок, чтобы 502 не спамил traceback

    application.run_polling(  # запускаем polling
        drop_pending_updates=True,  # не обрабатываем накопившиеся апдейты после долгого оффлайна
        # close_loop=False,  # не закрываем event loop принудительно (стабильнее на Windows)
    )  # конец run_polling



if __name__ == "__main__":
    main()
