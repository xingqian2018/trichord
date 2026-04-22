# Gateway API Clients (DEPRECATED)

> **Deprecated.** This file is kept for historical reference only.
> For current usage, refer to `UnifiedGatewayCalls.md` instead.

---

Async-backed clients for calling VLM and LLM models through the NVIDIA Gateway (OpenAI-compatible endpoint). Both classes share the same concurrency pattern: a dedicated asyncio event loop running in a background thread, with a semaphore to cap parallel requests.

---
## NvidiaGateway Model Choices

```python
MODEL_CHOICE = {
    "gemini-3-flash": "gcp/google/gemini-3-flash-preview",
    "gemini-3-pro": "gcp/google/gemini-3-pro",
}

RESPONSE_FORMAT = ["JSON", "YAML", ...]
```

## NvidiaGatewayVLM

Handles **vision-language model** calls with video/image input. Reads the API key from `credentials/nvidiagateway.secret` and connects to the NVIDIA inference endpoint via the `AsyncOpenAI` client.

**Key behaviors:**
- Concurrent requests controlled by `num_concurrency` (default 4)
- Per-request retry logic up to `num_max_retry` with timeout enforcement
- Streams responses and accumulates the full text before returning
- `build_request` formats the payload differently for Gemini vs. other models (system message placement)
- Supports JSON response format when `CAPTIONER_STRUCTURE_FORMAT` is set

```python
class NvidiaGatewayVLM:
    def __init__(self, num_concurrency: int = 4, num_max_retry: int = 1, timeout: int = 100):
        from openai import AsyncOpenAI
        with open("credentials/nvidiagateway.secret", "r") as f:
            api_key = f.read().strip()
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://inference-api.nvidia.com/v1")
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request: dict) -> str:
        for attempt in range(self.num_max_retry):
            try:
                async def _stream_response():
                    stream = await self.client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result

                return await asyncio.wait_for(_stream_response(), timeout=self.timeout)
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"VLM API timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s"
                    )
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"VLM API error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}"
                    )
        logger.warning("VLM query failed after max retries")
        return ""

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None)

        async def coroutine_gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def build_request(
        self,
        system_prompt: str,
        content_prompt: str,
        video_bytes: bytes,
        model_name: str,
        extension: str = "mp4",
    ) -> dict[str, Any]:
        video_url = video_bytes_to_data_url(video_bytes, extension)
        if model_name in ["gemini-3-pro"]:
            request = {
                "model": MODEL_CHOICE[model_name],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{system_prompt}\n{content_prompt}"},
                            {"type": "image_url", "image_url": {"url": video_url}},
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 32768,
                "stream": True,
            }
        else:
            request = {
                "model": MODEL_CHOICE[model_name],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content_prompt},
                            {"type": "image_url", "image_url": {"url": video_url}},
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 32768,
                "stream": True,
            }
        if RESPONSE_FORMAT in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()
```

---

## NvidiaGatewayLLM

Handles **text-only LLM** calls. Same concurrency architecture as `NvidiaGatewayVLM`, but `build_request` takes no video input and exposes `temperature` and `max_tokens` as parameters.

**Key differences from VLM:**
- No video/image encoding — text-only messages
- Configurable `temperature` and `max_tokens` per request (VLM hardcodes these)
- `query_core` breaks out of the retry loop on success (`break`), while VLM's `_query_one` returns immediately

```python
class NvidiaGatewayLLM:
    def __init__(self, num_concurrency=4, num_max_retry=1, timeout: int = 100):
        from openai import AsyncOpenAI

        with open("credentials/nvidiagateway.secret", "r") as f:
            self.api_key = f.read()
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://inference-api.nvidia.com/v1")
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request) -> str:
        feedback_one = ""
        for num_retry in range(self.num_max_retry):
            feedback_one = ""
            try:
                async def _stream_response():
                    stream = await self.client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result

                feedback_one = await asyncio.wait_for(_stream_response(), timeout=self.timeout)
                break
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if num_retry == 0 or num_retry == self.num_max_retry - 1:
                    logger.warning(
                        f"LLM API timeout (attempt {num_retry + 1}/{self.num_max_retry}): exceeded {self.timeout}s"
                    )
            except Exception as e:
                if num_retry == 0 or num_retry == self.num_max_retry - 1:
                    logger.warning(
                        f"LLM API error (attempt {num_retry + 1}/{self.num_max_retry}): {type(e).__name__}: {e}"
                    )
            if num_retry == self.num_max_retry - 1:
                logger.warning("NvidiaGatewayLLM generation failed by reaching max retries")
        return feedback_one

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None)

        async def coroutine_gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def _one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(_one(r) for r in request_list))

        r = asyncio.run_coroutine_threadsafe(coroutine_gather(), self._loop)
        result = r.result()
        pbar.close()
        return result

    def build_request(
        self,
        system_prompt: str,
        content_prompt: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        if "gemini-3-pro" in model_name:
            request = {
                "model": MODEL_CHOICE[model_name],
                "messages": [
                    {"role": "user", "content": f"{system_prompt}\n{content_prompt}"}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
        else:
            request = {
                "model": MODEL_CHOICE[model_name],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
        if RESPONSE_FORMAT in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()
```

## Lepton VLM:

```python
class LeptonGatewayLLM(NvidiaGatewayLLM):
    def __init__(self, num_concurrency=4, num_max_retry=1, timeout=45, gateway_url=""):
        from openai import AsyncOpenAI

        self.base_url = gateway_url
        api_key = os.environ.get("LEPTON_API_TOKEN")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()
```
