# poster_agent.py

import base64
import json
import time
import uuid
from pathlib import Path

from openai import OpenAI


client = OpenAI()

OUTPUT_DIR = Path("poster_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# The harness only sees image IDs.
# Your application owns the actual files.
IMAGE_STORE: dict[str, Path] = {}


# ============================================================
# Tool implementation 1: text2image
# ============================================================

def text2image(
    prompt: str,
    width: int = 1536,
    height: int = 2048,
) -> dict:
    """
    Generate a poster image.
    """

    response = client.images.generate(
        model="gpt-image-2.5-sunburst",
        prompt=prompt,
        size=f"{width}x{height}",
        quality="high",
        output_format="png",
    )

    image_bytes = base64.b64decode(response.data[0].b64_json)

    image_id = str(uuid.uuid4())
    path = OUTPUT_DIR / f"{image_id}.png"

    path.write_bytes(image_bytes)
    IMAGE_STORE[image_id] = path

    return {
        "image_id": image_id,
        "path": str(path),
    }


# ============================================================
# Tool implementation 2: poster_critic
# ============================================================

def poster_critic(
    image_id: str,
    original_request: str,
) -> dict:

    path = IMAGE_STORE[image_id]

    image_b64 = base64.b64encode(path.read_bytes()).decode()

    critic_prompt = f"""
You are a professional poster design critic.

Original user request:

{original_request}

Evaluate the provided poster.

Judge:
- whether it satisfies the user's request
- composition
- typography
- text correctness
- hierarchy
- readability
- visual quality
- spacing
- whether anything important is missing

Return ONLY JSON:

{{
  "pass": true or false,
  "score": integer 0-100,
  "problems": ["problem 1", "problem 2"],
  "revision_prompt": "specific instructions for improving the next poster"
}}

Use pass=true only if the poster is genuinely production quality.
"""

    response = client.responses.create(
        model="gpt-6-astra",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": critic_prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                        "detail": "original",
                    },
                ],
            }
        ],
    )

    text = response.output_text.strip()

    # In production I'd use Structured Outputs here.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "pass": False,
            "score": 0,
            "problems": ["Critic returned malformed JSON"],
            "revision_prompt": text,
        }


# ============================================================
# These are the TWO tools exposed to Codex harness
# ============================================================

TOOLS = [
    {
        "type": "function",
        "name": "text2image",
        "description": (
            "Generate a poster image from a detailed visual prompt. "
            "Returns an image_id that can be passed to poster_critic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed image-generation prompt.",
                },
                "width": {
                    "type": "integer",
                    "description": "Poster width in pixels.",
                },
                "height": {
                    "type": "integer",
                    "description": "Poster height in pixels.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },

    {
        "type": "function",
        "name": "poster_critic",
        "description": (
            "Visually inspect a generated poster and determine whether it "
            "satisfies the original request. Returns problems and specific "
            "revision instructions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "Image ID returned by text2image.",
                },
                "original_request": {
                    "type": "string",
                    "description": "The user's original poster request.",
                },
            },
            "required": [
                "image_id",
                "original_request",
            ],
            "additionalProperties": False,
        },
    },
]


# ============================================================
# Tool dispatcher
# ============================================================

def execute_tool(name: str, args: dict):

    print(f"\n>>> TOOL CALL: {name}")
    print(json.dumps(args, indent=2))

    if name == "text2image":
        result = text2image(**args)

    elif name == "poster_critic":
        result = poster_critic(**args)

    else:
        raise ValueError(f"Unknown tool: {name}")

    print(f">>> TOOL RESULT:")
    print(json.dumps(result, indent=2))

    return result


# ============================================================
# CODEX HARNESS
# ============================================================

AGENT_INSTRUCTIONS = """
You are an expert poster-design agent.

Your job is to turn the user's request into a production-quality poster.

You have two tools:

1. text2image
   Generates a poster from a detailed prompt.

2. poster_critic
   Visually evaluates a generated poster.

WORKFLOW:

1. Understand the user's request.
2. Design an excellent poster concept.
3. Write a detailed image-generation prompt.
4. Call text2image.
5. ALWAYS call poster_critic on the resulting image.
6. Carefully reason about the critic feedback.

If poster_critic returns pass=false:
    - improve the poster prompt using the critique
    - call text2image again
    - critic the new result again

You may attempt at most 3 generated versions.

If the critic returns pass=true, stop iterating.

When finished, report:
- final image_id
- final image path
- critic score
- short explanation of what you created

Do NOT claim the poster is good without calling poster_critic.
"""


def make_poster(user_request: str):

    # ----------------------------------------
    # THIS creates the managed Codex harness.
    # ----------------------------------------

    session = client.beta.agents.sessions.create(
        agent={
            "model": "gpt-6-astra",
            "instructions": AGENT_INSTRUCTIONS,
            "tools": TOOLS,
        },

        # We don't need Codex shell/files here.
        # Our tools execute in this Python process.
        environment={
            "type": "none",
        },

        input=user_request,
    )

    session_id = session.id

    print("SESSION:", session_id)

    # ------------------------------------------------
    # Your application services function calls.
    #
    # Reasoning / deciding WHICH tool to call is
    # performed by the Codex harness.
    # ------------------------------------------------

    while True:

        session = client.beta.agents.sessions.retrieve(
            session_id
        )

        # --------------------------------------------
        # HARNESS WANTS ONE OF OUR TOOLS
        # --------------------------------------------

        if session.status == "requires_action":

            results = []

            for action_obj in session.required_actions:

                action = action_obj.to_dict()

                if action["type"] != "function_call":
                    raise RuntimeError(
                        f"Unexpected action: {action}"
                    )

                try:
                    output = execute_tool(
                        action["name"],
                        action["arguments"],
                    )

                    result_event = {
                        "type":
                            "agent.session.input.tool_result",

                        "turn_id":
                            action["turn_id"],

                        "call_id":
                            action["call_id"],

                        "success":
                            True,

                        "output":
                            json.dumps(output),
                    }

                except Exception as e:

                    result_event = {
                        "type":
                            "agent.session.input.tool_result",

                        "turn_id":
                            action["turn_id"],

                        "call_id":
                            action["call_id"],

                        "success":
                            False,

                        "error":
                            str(e),
                    }

                results.append(result_event)

            # Give results back to Codex harness.
            #
            # It now resumes reasoning automatically.
            client.beta.agents.sessions.events.create(
                session_id,
                events=results,
            )

            continue

        # --------------------------------------------
        # STILL THINKING / RUNNING
        # --------------------------------------------

        if session.status == "in_progress":
            time.sleep(0.5)
            continue

        # --------------------------------------------
        # FAILED
        # --------------------------------------------

        if session.status == "failed":
            raise RuntimeError(session.error)

        # --------------------------------------------
        # FINISHED
        # --------------------------------------------

        if session.status == "idle":
            break

        time.sleep(0.5)

    # Retrieve everything the harness produced.
    items = client.beta.agents.sessions.items.list(
        session_id,
        order="asc",
        limit=100,
    )

    print("\n\n========== SESSION RESULT ==========\n")

    print(items.to_json(indent=2))

    return session_id


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    make_poster(
        """
Create a vertical research poster for a paper titled:

"Stable Signed Flow Reinforcement"

The visual metaphor should be a flow controller stabilizing
positive and negative reinforcement signals.

Style:
- modern AI research
- dark minimal background
- blue positive flow
- orange negative flow
- strong typography
- visually elegant
- suitable for an ICLR research teaser
- no unnecessary text
"""
    )