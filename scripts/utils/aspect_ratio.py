import math

ASPECT_RATIOS = ["16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16"]
ASPECT_RATIOS_V0 = ["1:1", "4:3", "3:4", "16:9", "9:16"]
TALL_BUCKET = "<=1:2"
WIDE_BUCKET = ">=2:1"
TALL_LIMIT = 0.5
WIDE_LIMIT = 2.0

RESOLUTION_TIERS = {
    "256p": 256 * 256,
    "480p": 854 * 480,
    "720p": 1280 * 720,
    "768p": 1024 * 1024,
    "1080p": 1920 * 1080,
    "1440p": 2560 * 1440,
    "1536p": 2048 * 2048,
    "2160p": 3840 * 2160,
}
RESOLUTION_RANGES = {
    "lt_256p": (0, 59904),
    "256p": (59904, 400384),
    "480p": (400384, 897024),
    "720p": (897024, 1027840),
    "768p": (1027840, 2030336),
    "1080p": (2030336, 3625472),
    "1440p": (3625472, 4139520),
    "1536p": (4139520, 8213760),
    "2160p": (8213760, None),
}
AREA_1K = RESOLUTION_TIERS["768p"]
AREA_2K = RESOLUTION_TIERS["1536p"]
ALIGN_1K = 16
ALIGN_2K = 32


def ratio_value(aspect_ratio: str) -> float:
    width, height = (int(x) for x in aspect_ratio.split(":"))
    return width / height


def ratio_edges(ratios: list[str] = ASPECT_RATIOS) -> list[tuple[str, float, float]]:
    ordered = sorted(ratios, key=ratio_value)
    values = [ratio_value(r) for r in ordered]
    edges = [math.sqrt(a * b) for a, b in zip(values, values[1:])]
    lows = [TALL_LIMIT] + edges
    highs = edges + [WIDE_LIMIT]
    return list(zip(ordered, lows, highs))


def aspect_ratio_bucket(width: int, height: int, ratios: list[str] = ASPECT_RATIOS) -> str:
    r = width / height
    if r <= TALL_LIMIT:
        return TALL_BUCKET
    if r >= WIDE_LIMIT:
        return WIDE_BUCKET
    return min(ratios, key=lambda target: abs(math.log(r / ratio_value(target))))


def resolution_tier(width: int, height: int) -> str:
    area = width * height
    for tier, (low, high) in RESOLUTION_RANGES.items():
        if area >= low and (high is None or area < high):
            return tier
    raise ValueError(f"no resolution tier for {width}x{height}")


def canvas_size(aspect_ratio: str, area: int, align: int) -> tuple[int, int]:
    r = ratio_value(aspect_ratio)
    return round(math.sqrt(area * r) / align) * align, round(math.sqrt(area / r) / align) * align


def size_text(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


SIZE_1K = {r: canvas_size(r, AREA_1K, ALIGN_1K) for r in ASPECT_RATIOS}
SIZE_2K = {r: canvas_size(r, AREA_2K, ALIGN_2K) for r in ASPECT_RATIOS}
