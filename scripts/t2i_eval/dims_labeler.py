import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from utils import GATEWAY_CONFIG, UnifiedGatewayLLM

MODELS = ("opus-5.5", "kimi-k3", "gpt-6-astra")
BASE_DIMS = ("Quality", "Aesthetics", "Alignment")
RWF = "Real-world Fidelity"
CG = "Creative Generation"

FACETS = {
    "Quality": {
        "Realism": {
            "Physical Logic": "adherence to physical laws (gravity, reflection, shadow direction, object stability)",
            "Material Texture": "realistic surface texture of materials (skin, fabric, metal, wood)",
        },
        "Detail": {
            "Noise": "rich detail without noise or unnatural smoothing",
            "Edge Clarity": "sharp, well-defined edges and outlines",
            "Naturalness": "no artificial plastic or greasy AI look",
        },
        "Resolution": {"Resolution": "high-definition, free from pixelation or compression artifacts"},
    },
    "Aesthetics": {
        "Composition": {"Composition": "balanced, visually guided, pleasing composition"},
        "Color Harmony": {"Color Harmony": "harmonious, cohesive palette fitting the mood"},
        "Lighting": {"Lighting & Atmosphere": "light and shadow atmosphere matching the scene"},
        "Anatomical Portraiture": {"Anatomical Fidelity": "correct facial proportions, skeleton, limbs, realistic skin micro-texture"},
        "Emotional Expression": {"Emotional Expression": "image tone conveys the intended emotion or mood"},
        "Style Control": {"Style Control": "accurately captures a requested artistic style"},
    },
    "Alignment": {
        "Attributes": {
            "Quantity": "number of objects matches the prompt",
            "Facial Expression": "face of a person or animal shows the specified emotion",
            "Material Properties": "object materials match the description",
            "Color": "object colors match the specification",
            "Shape": "object shapes match the description",
            "Size": "object sizes match the specification",
        },
        "Actions": {
            "Contact Interaction": "physical contact between subjects",
            "Non-contact Interaction": "non-contact spatial or social relationship between subjects",
            "Full-body Action": "posture and body action of a person or animal performing an activity",
        },
        "Layout": {
            "2D Space": "left/right, top/bottom, foreground/background positions",
            "3D Space": "3D layout, occlusion, relative position in depth",
        },
        "Relations": {
            "Composition Relationship": "integrating multiple elements into a coherent whole",
            "Difference/Similarity": "specified differences or similarities between objects",
            "Containment": "containment or enclosure relationships",
        },
        "Scene": {
            "Real-world Scene": "scene type and environment (office, forest, street)",
            "Virtual Scene": "internal consistency of a fictional or fantasy scene",
        },
    },
    RWF: {
        "Fairness": {
            "Social Bias": "avoid tying genders to particular professions or settings",
            "Cultural Fairness": "avoid stereotypical portrayal by region, race or culture",
        },
        "Safety & Compliance": {"Safety & Compliance": "avoid prohibited content (pornography, violence, hate symbols)"},
        "World Knowledge": {
            "Animals": "real animals with anatomically accurate features and biology",
            "Objects": "typical appearance, structure, brand logo or iconic look of real-world items",
            "Information Visualization": "translating an abstract or scientific concept into an understandable visual",
            "Temporal Characteristics": "iconic elements of a specific historical period (technology, clothing, architecture)",
            "Cultural Elements": "symbols, traditional clothing, rituals, customs depicted accurately",
        },
    },
    CG: {
        "Imagination": {"Imagination": "creative, imaginative combination of novel or surreal elements"},
        "Feature Matching": {"Feature Matching": "seamless fusion of multiple elements"},
        "Logical Resolution": {"Logical Resolution": "causal relationships between events (breaking glass, shards flying)"},
        "Text Rendering": {
            "Text Accuracy": "text in the image is clear, legible, spelled correctly",
            "Text Layout": "layout of text (alignment, spacing, margins)",
            "Font": "a specified font or typeface",
            "Cross-lingual Generation": "translation or text in another language",
        },
        "Design Applications": {
            "Graphic Design": "advertisement, poster, information hierarchy",
            "Product Design": "industrial design logic of a product",
            "Spatial Design": "interior or architectural space design",
            "Fashion Styling": "clothing cut, silhouette, makeup style",
            "Game Design": "game props and UI elements",
            "Art Design": "a specific artistic design style",
        },
        "Visual Storytelling": {
            "Cinematic Style": "a named director's visual language",
            "Camera / Lens Style": "specific photographic equipment or lens look",
            "Storyboard Creation": "panel layout (multi-panel, split-screen)",
            "Shot Sizes": "close-up, medium shot, wide shot",
            "Composition": "composition rules (thirds, golden ratio, leading lines)",
            "Angles": "bird's-eye, low angle, Dutch angle",
            "Comic Creation": "comic style (American comics, manga, BD)",
        },
    },
}

FACET_TO_L1 = {f"{l1} / {l2} / {l3}": l1 for l1, l2s in FACETS.items() for l2, l3s in l2s.items() for l3 in l3s}

BASELINE_PROMPT = """You label text-to-image benchmark prompts for a judge that scores images on five dimensions. Quality, Aesthetics and Alignment are always scored. Decide whether the two optional dimensions apply to the prompt.

Real-world Fidelity applies when the prompt's image must be faithful to real-world knowledge or norms:
- Fairness: depicting people in roles, professions or settings where social bias or cultural stereotyping could appear.
- Safety & Compliance: content near safety boundaries.
- World Knowledge: real animal species, real-world objects with a typical appearance, structure, brand or iconic characteristics, scientific or abstract concepts to visualize, a specific historical period, or cultural elements (symbols, traditional clothing, rituals, customs).

Creative Generation applies when the prompt asks for creative or professional-design output:
- Imagination (novel or surreal combinations), Feature Matching (fusing multiple elements), Logical Resolution (depicting cause and effect).
- Text Rendering: text in the image, its layout, a font, or cross-lingual text.
- Design Applications: graphic design (ads, posters), product design, spatial/interior/architectural design, fashion styling, game design, a specific art design style.
- Visual Storytelling: a director's cinematic style, a camera or lens look, storyboard panels, a shot size, a composition rule, a camera angle, comic style.

Respond with a JSON object only: {"real_world_fidelity": true|false, "creative_generation": true|false}"""

STRICT_PROMPT = """You label text-to-image benchmark prompts for a judge that scores each generated image on five dimensions. Quality, Aesthetics and Alignment are always scored. You decide whether the two optional dimensions are worth scoring for this specific prompt. Most prompts are plain scene descriptions where neither applies, and roughly half of all prompts get each label, so be selective: answer true only when the prompt itself contains an explicit trigger listed below, never merely because the topic could be related.

real_world_fidelity = true only if scoring the image requires checking it against real-world facts, i.e. the prompt contains at least one of:
- a named real animal, insect or plant species, or a biological detail a viewer could judge for correctness;
- a recognizable real-world object, product, brand, landmark, vehicle, instrument or device whose true appearance or structure matters;
- a specific culture, ethnicity, nationality, religion, tradition, festival, ritual, traditional garment or regional architecture;
- a specific historical era or period-accurate technology, clothing or architecture;
- a scientific, technical or abstract concept, chart, diagram or infographic that must be visualized correctly;
- a person described by profession, gender, age or ethnicity in a way where stereotyping could occur;
- content near safety boundaries (violence, weapons, nudity, hateful symbols).
Generic everyday scenes (a room, a street, a cup, a person holding an everyday item, a landscape, light and shadow, composition, color, a generic person or pet) are false.

creative_generation = true only if the prompt contains at least one of:
- text that must appear in the image (a quoted title, label, slogan, sign, caption, number), a named font, or a translation request;
- a design deliverable: advertisement, poster, packaging, logo, UI, product or industrial design, interior or architectural design, fashion or makeup design, game asset or prop sheet, a named art or design style;
- a cinematic or photographic craft instruction: a named director, a specific camera or lens, multi-panel storyboard, comic or manga layout, an explicit shot size, camera angle or composition rule;
- a surreal, fantasy or impossible combination of elements, a fusion of several unrelated elements, or an explicit cause-and-effect event to depict (something breaking, spilling, collapsing, reacting).
A realistic scene with ordinary details, even a beautifully lit or detailed one, is false.

Respond with a JSON object only: {"real_world_fidelity": true|false, "creative_generation": true|false}"""


def facet_listing() -> str:
    lines = []
    for l1, l2s in FACETS.items():
        lines.append(f"[{l1}]")
        for l2, l3s in l2s.items():
            for l3, desc in l3s.items():
                lines.append(f"- {l1} / {l2} / {l3}: {desc}")
    return "\n".join(lines)


FACET_PROMPT = (
    """You prepare the evaluation checklist for a text-to-image judge. The judge scores a generated image against a fixed taxonomy of 56 facets (L1 / L2 / L3). Given only the prompt, choose which facets are actually examined by this prompt, then the judge scores only the L1 groups that your chosen facets belong to.

Taxonomy:
"""
    + facet_listing()
    + """

Rules:
- Choose between 3 and 10 facets, usually around 5. Pick a facet only when the prompt explicitly asks for, describes or strongly depends on what that facet checks; do not add facets just because they could loosely apply.
- Quality, Aesthetics and Alignment facets are about the image itself; pick the ones that this prompt visibly stresses.
- Real-world Fidelity facets are chosen only when judging the image needs outside real-world knowledge or norms: a real species, a real branded or iconic object, a specific culture or historical era, a concept or infographic to be visualized correctly, a person whose profession or origin could invite stereotypes, or safety-sensitive content. Generic scenes, generic people and generic objects do not qualify.
- Creative Generation facets are chosen only when the prompt contains a creative or professional-design demand: text to be rendered, a named font or translation, an ad/poster/product/interior/fashion/game/art-style design task, cinematic or camera instructions (director, lens, storyboard, comic, shot size, angle, composition rule), or surreal, fused or cause-and-effect content. Realistic scenes with rich detail do not qualify.
- Use the exact facet names as written above.

Respond with a JSON object only: {"facets": ["<L1> / <L2> / <L3>", ...]}"""
)

FACET_PROMPT_V2 = FACET_PROMPT.replace(
    "- Use the exact facet names as written above.",
    "- Text Accuracy, Text Layout, Font and Cross-lingual Generation are chosen only when words, letters or numbers must appear as visible text in the image; Cross-lingual Generation additionally needs a translation request or text in a non-English script.\n"
    "- Shot Sizes, Angles, Composition and Camera / Lens Style are chosen only when the framing, viewpoint, composition rule or equipment is itself a stated requirement of the prompt; an incidental word such as close-up or wide in an ordinary scene description is not enough.\n"
    "- Feature Matching, Imagination and Logical Resolution are chosen only for clearly fused, surreal or cause-and-effect content.\n"
    "- Use the exact facet names as written above.",
)

SCHEMES = {
    "baseline": {"system": BASELINE_PROMPT, "kind": "binary"},
    "strict": {"system": STRICT_PROMPT, "kind": "binary"},
    "facets": {"system": FACET_PROMPT, "kind": "facets"},
    "facets2": {"system": FACET_PROMPT_V2, "kind": "facets"},
}

FINAL_RULES = {"rwf": "any", "cg": "majority"}

USER_TEMPLATE = "Prompt: {prompt}"

RULES = {
    "majority": lambda n, total: n * 2 > total,
    "any": lambda n, total: n >= 1,
    "all": lambda n, total: n == total,
}


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text[text.rfind("</think>") + len("</think>") :] if "</think>" in text else text
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except ValueError:
                return None
    return None


def parse_vote(kind: str, text: str) -> Optional[dict]:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return None
    if kind == "binary":
        if isinstance(parsed.get("real_world_fidelity"), bool) and isinstance(parsed.get("creative_generation"), bool):
            return {"real_world_fidelity": parsed["real_world_fidelity"], "creative_generation": parsed["creative_generation"]}
        return None
    facets = parsed.get("facets")
    if not isinstance(facets, list):
        return None
    kept = []
    for f in facets:
        if isinstance(f, str):
            f = re.sub(r"\s*/\s*", " / ", f.strip())
            if f in FACET_TO_L1 and f not in kept:
                kept.append(f)
    if not kept:
        return None
    l1s = {FACET_TO_L1[f] for f in kept}
    return {"real_world_fidelity": RWF in l1s, "creative_generation": CG in l1s, "facets": kept}


def vote_key(scheme: str, model: str, prompt: str) -> str:
    return hashlib.sha1(f"{scheme}\x00{model}\x00{SCHEMES[scheme]['system']}\x00{prompt}".encode()).hexdigest()


def aggregate(votes: dict[str, dict], rules: dict[str, str]) -> dict:
    total = len(votes)
    dims = list(BASE_DIMS)
    n_rwf = sum(v["real_world_fidelity"] for v in votes.values())
    n_cg = sum(v["creative_generation"] for v in votes.values())
    if RULES[rules["rwf"]](n_rwf, total):
        dims.append(RWF)
    if RULES[rules["cg"]](n_cg, total):
        dims.append(CG)
    out = {"dims": "; ".join(dims), "dims_votes": {m: {k: v[k] for k in ("real_world_fidelity", "creative_generation")} for m, v in votes.items()}}
    if any("facets" in v for v in votes.values()):
        counts: dict[str, int] = {}
        for v in votes.values():
            for f in v.get("facets", []):
                counts[f] = counts.get(f, 0) + 1
        out["dims_facets"] = [f for f in FACET_TO_L1 if counts.get(f, 0) * 2 > total]
    return out


class DimsLabeler:
    def __init__(
        self,
        scheme: str = "facets2",
        rules: Optional[dict[str, str]] = None,
        cache_path: Optional[Path] = None,
        num_concurrency: int = 16,
        timeout: int = 300,
        models: tuple[str, ...] = MODELS,
        max_rounds: int = 6,
        llm: Optional[UnifiedGatewayLLM] = None,
    ):
        self.scheme = scheme
        self.rules = rules or FINAL_RULES
        self.models = models
        self.max_rounds = max_rounds
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["vote"]
        self.owns_llm = llm is None
        self.llm = llm or UnifiedGatewayLLM(
            gateway_configs={"nvidia": GATEWAY_CONFIG["nvidia"]}, num_concurrency=num_concurrency, timeout=timeout
        )

    def request(self, scheme: str, model: str, prompt: str) -> dict:
        return self.llm.build_request(
            model_name=model,
            system_prompt=SCHEMES[scheme]["system"],
            content_prompt=USER_TEMPLATE.format(prompt=prompt),
            temperature=1.0,
            max_tokens=16384,
        )

    def store(self, key: str, vote: dict) -> None:
        self.cache[key] = vote
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "vote": vote}, ensure_ascii=False) + "\n")

    def collect(self, prompts: list[str], scheme: Optional[str] = None, models: Optional[tuple[str, ...]] = None) -> None:
        scheme = scheme or self.scheme
        models = models or self.models
        kind = SCHEMES[scheme]["kind"]
        for round_idx in range(self.max_rounds):
            todo = [(p, m) for p in dict.fromkeys(prompts) for m in models if vote_key(scheme, m, p) not in self.cache]
            if not todo:
                return
            logger.info(f"{scheme} round {round_idx}: {len(todo)} calls pending")
            responses = self.llm.query([self.request(scheme, m, p) for p, m in todo], pbar_desc=f"dims {scheme} r{round_idx}")
            failed = 0
            for (p, m), text in zip(todo, responses):
                vote = parse_vote(kind, text)
                if vote is None:
                    failed += 1
                else:
                    self.store(vote_key(scheme, m, p), vote)
            if failed:
                time.sleep(20)
        missing = sum(1 for p in prompts for m in models if vote_key(scheme, m, p) not in self.cache)
        if missing:
            raise RuntimeError(f"{missing} dims votes still missing after {self.max_rounds} rounds")

    def votes_for(self, prompt: str, scheme: Optional[str] = None, models: Optional[tuple[str, ...]] = None) -> dict[str, dict]:
        scheme = scheme or self.scheme
        return {m: self.cache[vote_key(scheme, m, prompt)] for m in (models or self.models)}

    def label(self, prompts: list[str]) -> list[dict]:
        self.collect(prompts)
        return [aggregate(self.votes_for(p), self.rules) for p in prompts]

    def close(self) -> None:
        if self.owns_llm:
            self.llm.close()
