# Gateway API Clients

Async-backed clients for calling VLM and LLM models through the NVIDIA and Lepton Gateway (OpenAI-compatible endpoint).
Both classes share the same concurrency pattern: a dedicated asyncio event loop running in a background thread, with a semaphore to cap parallel requests.

---

## Global Config

Shared by both `UnifiedGatewayVLM` and `UnifiedGatewayLLM`.

```python
GATEWAY_CONFIG = {
    "nvidia": {
        "url": "https://inference-api.nvidia.com/v1",
        "api": open("credentials/nvidiagateway.secret").read().strip(),
        "modelstr": [
            "openai/openai/gpt-5.4",
            "gcp/google/gemini-3.1-pro-preview",
            "nvidia/qwen/qwen-235b",
            "nvidia/qwen/qwen3-5-397b-a17b"
        ],
    },
    "lepton_qwen3_vl_235b_tr": {
        "url": "https://b5k2m9x7-qwen3-vl-235b-a22b-instruct-tr.xenon.lepton.run/v1",
        "api": os.environ["LEPTON_API_QWEN3_VL_235B"],
        "modelstr": ["LEPTON-TR/Qwen3-VL-235B-A22B-Instruct"],
    },
    "lepton_qwen3p5_397b_tr": {
        "url": "https://b5k2m9x7-qwen3p5-397b-a17b-tr.xenon.lepton.run/v1/",
        "api": os.environ["LEPTON_API_QWEN3P5_397B"],
        "modelstr": ["LEPTON-TR/Qwen3.5-397B-A17B"],
    },
}

MODEL_CHOICE = {
    "gemini-3.1-pro":               "gcp/google/gemini-3.1-pro-preview",
    "gpt-5.4":                      "openai/openai/gpt-5.4",
    "qwen-235b":                    "nvidia/qwen/qwen-235b",
    "qwen3-vl-235b-a22b-instruct":  "LEPTON-TR/Qwen3-VL-235B-A22B-Instruct",
    "qwen3.5-397b-a17b":            ["nvidia/qwen/qwen3-5-397b-a17b", "LEPTON-TR/Qwen3.5-397B-A17B"],
}

RESPONSE_FORMAT = ["JSON", "YAML", ...]
```

---

## VLM

### Helper functions:

```python
def image_bytes_to_data_url(image_bytes: bytes, image_fmt: str = "webp") -> str:
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime_type = mime_types.get(image_fmt.lower(), f"image/{image_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def video_bytes_to_data_url(video_bytes: bytes, video_fmt: str = "mp4") -> str:
    mime_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "avi": "video/avi",
    }
    mime_type = mime_types.get(video_fmt.lower(), f"video/{video_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(video_bytes).decode('utf-8')}"
```

### Core Gateway VLM Class:

```python
class UnifiedGatewayVLM:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=100):
        from openai import AsyncOpenAI
        self.key_to_gateway: dict[str, AsyncOpenAI] = dict()
        self.modelstr_to_key: dict[str, str] = dict()
        for keyname, cfg in gateway_configs.items():
            self.key_to_gateway[keyname] = AsyncOpenAI(api_key=cfg["api"], base_url=cfg["url"])
            for modelstr in cfg["modelstr"]:
                self.modelstr_to_key[modelstr] = keyname
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request: dict) -> str:
        try:
            client = self.key_to_gateway[self.modelstr_to_key[request["model"]]]
        except:
            return ""
        for attempt in range(self.num_max_retry):
            try:
                async def _stream_response():
                    stream = await client.chat.completions.create(**request)
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
                        f"VLM API for {request['model']} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s"
                    )
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"VLM API for {request['model']} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}"
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
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        image_bytes: Optional[bytes] = None,
        image_fmt: str = "webp",
        video_bytes: Optional[bytes] = None,
        video_fmt: str = "mp4",
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ) -> dict[str, Any]:

        if model_name in [""]:
            request = {}
        else:
            modelstr = MODEL_CHOICE[model_name]
            request = {
                "model": random.choice(modelstr) if isinstance(modelstr, list) else modelstr,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content_prompt},
                        ],
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

        extra_session_list = []

        if image_bytes is not None:
            image_url = image_bytes_to_data_url(image_bytes, image_fmt)
            extra_session_list.append({"type": "image_url", "image_url": {"url": image_url}})

        if video_bytes is not None:
            video_url = video_bytes_to_data_url(video_bytes, video_fmt)
            extra_session_list.append({"type": "image_url", "image_url": {"url": video_url}})

        if extra_session_list:
            request["messages"][-1]["content"].extend(extra_session_list)

        if output_fmt in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()

```

---

## LLM

### Core Gateway LLM Class:

```python
class UnifiedGatewayLLM:
    def __init__(self, gateway_configs: dict[str, dict], num_concurrency=4, num_max_retry=1, timeout=100):
        from openai import AsyncOpenAI
        self.key_to_gateway: dict[str, AsyncOpenAI] = dict()
        self.modelstr_to_key: dict[str, str] = dict()
        for keyname, cfg in gateway_configs.items():
            self.key_to_gateway[keyname] = AsyncOpenAI(api_key=cfg["api"], base_url=cfg["url"])
            for modelstr in cfg["modelstr"]:
                self.modelstr_to_key[modelstr] = keyname
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()

    async def query_core(self, request: dict) -> str:
        try:
            client = self.key_to_gateway[self.modelstr_to_key[request["model"]]]
        except:
            return ""
        for attempt in range(self.num_max_retry):
            try:
                async def _stream_response():
                    stream = await client.chat.completions.create(**request)
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
                        f"LLM API for {request['model']} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s"
                    )
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"LLM API for {request['model']} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}"
                    )
        logger.warning("LLM query failed after max retries")
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
        model_name: str,
        system_prompt: str,
        content_prompt: str,
        output_fmt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:

        if model_name in [""]:
            request = {}
        else:
            modelstr = MODEL_CHOICE[model_name]
            request = {
                "model": random.choice(modelstr) if isinstance(modelstr, list) else modelstr,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

        if output_fmt in ["JSON", "json"]:
            request["response_format"] = {"type": "json_object"}
        return request

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._t.join()

```
