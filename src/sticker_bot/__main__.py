from __future__ import annotations

import asyncio

from sticker_bot.app import main as app_main


def main() -> None:
    asyncio.run(app_main())


if __name__ == "__main__":
    main()
