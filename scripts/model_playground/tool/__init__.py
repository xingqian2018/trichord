from .generate_image import generate_image
from .poster_visual_critic_agent import poster_visual_critic_agent
from .render_poster import render_poster

TOOL_REGISTRY = {cls.SCHEMA["function"]["name"]: cls for cls in [generate_image, poster_visual_critic_agent, render_poster]}
