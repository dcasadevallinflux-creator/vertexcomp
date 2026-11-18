"""AI image generation (SDXL family) and the Versions node.

The C++ core only shows cached images (see core/src/ops/ai.cpp); generation
runs in a separate worker process (worker.py) driven by client.AIManager.
"""

GEN_TYPES = ("SDXLGeneration", "SDXLLightning", "LCMTurbo", "SDXLTurbo")
AI_TYPES = GEN_TYPES + ("Versions",)
# Knobs managed by the AI panels rather than shown as plain rows.
HIDDEN_KNOBS = {"image", "genInfo", "versions", "current"}
