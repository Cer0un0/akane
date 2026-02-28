from __future__ import annotations

import asyncio
from datetime import UTC, datetime


async def run_scheduler_loop():
    while True:
        print(f"[scheduler] tick {datetime.now(tz=UTC).isoformat()}")
        # TODO: enqueue maintenance.purge_expired and heartbeat jobs.
        await asyncio.sleep(60)


def main():
    asyncio.run(run_scheduler_loop())


if __name__ == "__main__":
    main()

