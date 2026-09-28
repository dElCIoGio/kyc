import asyncio

from checker import AGTNIFVerifier


async def main():
    async with AGTNIFVerifier(
        headless=False,
        timeout_ms=20_000,
    ) as verifier:

        result = await verifier.verify(
            ""
        )

        print(result)
        print("Verified:", result.verified)
        print("Name:", result.name)


asyncio.run(main())