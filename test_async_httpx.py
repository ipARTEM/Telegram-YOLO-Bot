# OD\test_async_httpx.py
import asyncio
import httpx


TOKEN = "7697971018:AAHgT4VuiNNxlwAhAl6NUOe8l5WojJwAJvY"


async def main() -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"
    timeout = httpx.Timeout(connect=30.0, read=30.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        print("STATUS:", response.status_code)
        print("TEXT:", response.text)


if __name__ == "__main__":
    asyncio.run(main())