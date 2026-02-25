# Telegram Bot for Puzzle Stickers (SAM2 + BiRefNet)
![Uploading image.png…]()

## Что делает бот
Бот режет фото/видео/GIF на сетку и создает Telegram sticker set из тайлов 100x100.

## Требования
- Windows 10/11
- Python 3.10+
- ffmpeg в `PATH`
- Telegram bot token
- (желательно) NVIDIA GPU + CUDA

## Быстрый старт (Windows PowerShell)

```powershell
git clone https://github.com/Rusti3/Tg_bot_for_stickers_with_sam2.git
cd Tg_bot_for_stickers_with_sam2

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Установить ffmpeg (если не установлен):

```powershell
winget install --id Gyan.FFmpeg -e
```

Установить зависимости:

```powershell
pip install -r requirements.txt
```

Если `pip` ругается на `torch==...+cu126`, установи torch из официального индекса и повтори:

```powershell
pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Настройка `.env`

```powershell
Copy-Item .env.example .env
```

Открой `.env` и укажи токен без лишних пробелов:

```env
BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
```

## Скачивание веса SAM2.1 (обязательно)

Создать папку и скачать `sam2.1_hiera_base_plus.pt`:

```powershell
New-Item -Path .\checkpoints -ItemType Directory -Force | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt?download=true" -OutFile ".\checkpoints\sam2.1_hiera_base_plus.pt"
Get-Item .\checkpoints\sam2.1_hiera_base_plus.pt
```

Источник веса:
- https://huggingface.co/facebook/sam2.1-hiera-base-plus/blob/main/sam2.1_hiera_base_plus.pt

## Запуск

```powershell
python .\birefnet_sam2_full.py
```

## Команды Telegram-бота

- `/stats`
  - Показывает количество пользователей и запросов.

- `/add [w_count] [back=...]`
  - `w_count`: ширина сетки от `1` до `10`.
  - `back=...`: режим фона.

Рабочие примеры:

```text
/add
/add 3
/add 4 back=auto
/add 5 back=black30
/add 2 back=#ffffff20
```

Поддерживаемые типы входа:
- Фото
- Видео
- Анимация
- GIF (как document `.gif`)
- Ответ на сообщение с медиа

Поддерживаемые `back=`:
- `auto` — авто-маска (BiRefNet/SAM2)
- `none` — без удаления фона
- Именованный цвет + допуск: `black30`, `white20`, `red40`, `green25`, `blue35`
- HEX + допуск: `#ffffff20`, `#00000030`

