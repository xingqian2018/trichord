#!/usr/bin/env python3
"""Text-to-image batch client for the Cosmos3 text2image_server API.

Submits jobs via ``POST /generate``, polls via ``GET /jobs/{id}``, and decodes
``image_data`` (base64) directly from the JSON response.

Run from anywhere:
    export T2I_ENDPOINT_URL=https://<your-endpoint>.lepton.run
    export T2I_AUTH_TOKEN=<your-token>
    python3 lepton_call_t2i_distill.py
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import aiohttp

MAX_CONNECTIONS = 4  # semaphore: max concurrent live jobs

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "out"

JOBS: list[str] = [
    "A golden retriever sitting in a sunny park, photorealistic.",
    "An astronaut walking on the surface of Mars at sunset.",
    "A cozy coffee shop interior with rain on the windows.",
    "A classical Chinese ink wash painting with a profound and distant artistic conception. In the distance, the grand silhouette of the Egyptian pyramids is lightly rendered with pale ink, standing in the vast sea of sand, with concise brushstrokes and a desolate mood. The main subject of the painting is two adorable giant pandas. They are wearing slightly cumbersome white spacesuits, revealing a leisurely expression under their transparent round helmets. One is sitting, while the other is leaning against a sand dune, both relishing the fresh, verdant green bamboo in their hands, with the bamboo leaves sketched with lively strokes. The overall style is freehand, with well-balanced shades of ink. The combination of modern technology and classical charm forms a bizarre yet harmonious and wonderful composition on the rice paper.",
    "In a warm and healing picture book style, a young gardener wearing a straw hat and work overalls is squatting in a dense, colorful flower bed. He is wearing thick canvas gloves, carefully holding up a sunflower that is bent over because its flower head is too large. This sunflower has full, golden-yellow petals and is slightly bowing its head. The warm afternoon sun casts soft speckles of light through the gaps in the leaves, illuminating the gardener's focused and gentle expression. The entire scene is filled with tranquility and vitality.",
    "A close-up shot focuses on a textured wooden chessboard. In the center of the frame, a black king chess piece stands proudly, carved with a smugly smiling face, its mouth corners turned up. Beside it, a white king chess piece lies powerlessly on the chessboard square, also carved with a sad expression, its eyes downcast. Dramatic lighting illuminates the victor, while the loser is half-hidden in shadow. The surrounding chess pieces are blurred out of focus, suggesting a fierce duel that has just ended, the outcome decided, and the game settled.",
]

POLL_INTERVAL_S = 20.0
REQUEST_TIMEOUT_S = 120.0


def _headers(auth_token: str) -> dict:
    return {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
    }


async def _submit(session: aiohttp.ClientSession, base_url: str, prompt: str) -> tuple[str, str | None]:
    payload = {"prompt": prompt, "prompt_upsampling": True}
    async with session.post(f"{base_url}/generate", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Bad submit response: {data}")
    replica = data.get("lepton_replica_id") or None
    return str(job_id), replica


async def _poll(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    idx: int,
    timeout: float = 3600.0,
) -> dict:
    url = f"{base_url}/jobs/{job_id}"

    async def _loop() -> dict:
        while True:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
            state = data.get("status")
            print(f"[{idx:02d}] [{state}] [job={job_id[:8]}]", flush=True)
            if state == "completed":
                return data
            if state == "failed":
                raise RuntimeError(f"Job failed: {data.get('error')}")
            await asyncio.sleep(POLL_INTERVAL_S)

    return await asyncio.wait_for(_loop(), timeout=timeout)


async def process_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    idx: int,
    prompt: str,
) -> tuple[int, str | None]:
    output_path = OUTPUT_DIR / f"{idx:02d}.png"
    if output_path.exists():
        print(f"[{idx:02d}] [skip] exists {output_path.name}", flush=True)
        return idx, None

    async with semaphore:
        try:
            print(f"[{idx:02d}] [submitting] {prompt[:60]}...", flush=True)
            job_id, _ = await _submit(session, base_url, prompt)
            print(f"[{idx:02d}] [submitted] [job={job_id[:8]}]", flush=True)

            result = await _poll(session, base_url, job_id, idx)

            image_data_b64 = result.get("image_data")
            if not isinstance(image_data_b64, str) or not image_data_b64:
                raise RuntimeError(f"No image_data in response: {result}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(image_data_b64))
            print(f"[{idx:02d}] [saved] output {output_path.name}", flush=True)
            return idx, None
        except Exception as exc:
            print(f"[{idx:02d}] [error] {exc}", flush=True)
            return idx, str(exc)


async def main_async() -> None:
    endpoint_url = os.environ.get("T2I_ENDPOINT_URL", "").strip()
    auth_token = os.environ.get("T2I_AUTH_TOKEN", "").strip()
    if not endpoint_url:
        raise SystemExit("Set T2I_ENDPOINT_URL to the text2image server base URL.")
    if not auth_token:
        raise SystemExit("Set T2I_AUTH_TOKEN to the endpoint bearer token.")

    base_url = endpoint_url.rstrip("/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Endpoint:    {endpoint_url}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"Jobs:        {len(JOBS)}  Max concurrent: {MAX_CONNECTIONS}")

    semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)

    async with aiohttp.ClientSession(headers=_headers(auth_token), timeout=timeout) as session:
        tasks = [
            process_one(session, semaphore, base_url, idx, prompt)
            for idx, prompt in enumerate(JOBS)
        ]
        results = await asyncio.gather(*tasks)

    errors = [(idx, err) for idx, err in results if err]
    print(f"\nDone. {len(JOBS) - len(errors)}/{len(JOBS)} succeeded.")
    if errors:
        print(f"Failed ({len(errors)}):")
        for idx, err in sorted(errors):
            print(f"  [{idx:02d}] {err}")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
