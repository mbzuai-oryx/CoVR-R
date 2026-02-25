DESCRIPTION_GUIDELINES = (
    "Focus your description on the observable video content. Cover the primary "
    "subjects, their actions and transitions, the scene environment and noteworthy "
    "background elements, the lighting and color palette, the camera motion or "
    "framing, the pacing of events, and the overall atmosphere or mood. "
    "Write in the present tense as a single cohesive paragraph without bullet points."
)


def _kw_reference():
    return (
        "Extract key visual elements from this video as a list of keywords. "
        "Include only the most important elements: main subjects, key actions, "
        "scene type, lighting style, camera movement. "
        'Format as comma-separated keywords (e.g., "person walking, urban street, '
        'daytime, handheld camera, busy atmosphere"). '
        "Keep it concise - 8-12 keywords maximum."
    )

def _kw_edit(edit_instruction):
    return (
        "Apply the following edit instructions to the reference video and "
        "extract key visual elements from the edited video.\n\n"
        f"Edit instructions:\n{edit_instruction}\n\n"
        "List the key visual elements as comma-separated keywords focusing on: "
        "main subjects, key actions, scene type, lighting style, camera movement, "
        "atmosphere.\nKeep it concise - 8-12 keywords maximum.\n"
        "Reflect the changes from the edit instruction in the keywords."
    )

def _kw_edit_reasoning(edit_instruction):
    return (
        f'Analyze how this edit instruction changes the reference video: "{edit_instruction}"\n\n'
        "Identify in 2-3 sentences:\n"
        "1. What specific visual elements are being modified or replaced\n"
        "2. The key visual differences between before and after the edit\n\n"
        "Be concrete and focus only on the changed elements, not unchanged background details."
    )

def _kw_edit_description(edit_instruction, reasoning_trace):
    return (
        "Extract key visual elements from the edited video as a list of keywords.\n\n"
        f'Edit applied: "{edit_instruction}"\n\n'
        f"Based on this analysis:\n{reasoning_trace}\n\n"
        "List the key visual elements as comma-separated keywords focusing on: "
        "main subjects, key actions, scene type, lighting style, camera movement, "
        "atmosphere.\nKeep it concise - 8-12 keywords maximum.\n"
        "Reflect the changes from the edit instruction in the keywords."
    )


def _dense_reference():
    return f"Describe the contents of this reference video in vivid detail. {DESCRIPTION_GUIDELINES}"

def _dense_edit(edit_instruction):
    return (
        "Apply the following edit instructions to the reference video and "
        f"describe the resulting edited video in vivid detail.\n{DESCRIPTION_GUIDELINES}\n"
        f"Edit instructions:\n{edit_instruction}"
    )

def _dense_edit_reasoning(edit_instruction):
    return (
        "Given this reference video, analyze the following edit instruction "
        "and reason about how the video would change.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Provide a concise reasoning trace that covers:\n"
        "1. What key elements in the reference video would be affected\n"
        "2. What specific transformations would occur\n"
        "3. What the result would look like after applying the edit\n\n"
        "Keep your reasoning focused and concrete."
    )

def _dense_edit_description(edit_instruction, reasoning_trace):
    return (
        "Based on the reasoning below, describe the edited video in vivid detail.\n"
        f"{DESCRIPTION_GUIDELINES}\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Reasoning:\n{reasoning_trace}\n\n"
        "Now provide a detailed description of the edited video:"
    )


def _reason_reference():
    return f"Describe the contents of this reference video in vivid detail. {DESCRIPTION_GUIDELINES}"

def _reason_edit(edit_instruction):
    return f"Describe the video after this edit: {edit_instruction}"

def _reason_edit_reasoning(edit_instruction):
    return (
        "Given this reference video and the edit instruction below, "
        "briefly think through what the edited video would look like.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Describe only the key changes that would occur:"
    )

def _reason_edit_description(edit_instruction, reasoning_trace):
    return (
        f"You previously identified these changes: {reasoning_trace}\n\n"
        "Now, apply the edit instruction to the reference video and "
        f"describe the resulting edited video in vivid detail.\n{DESCRIPTION_GUIDELINES}\n\n"
        f"Edit instruction:\n{edit_instruction}"
    )


def _default_synthesis(edit_instruction, reasoning_traces):
    traces_text = "\n\n".join(
        [f"Analysis {i+1}: {trace}" for i, trace in enumerate(reasoning_traces)]
    )
    return (
        f'Edit instruction: "{edit_instruction}"\n\n'
        f"Multiple analyses:\n{traces_text}\n\n"
        "Synthesize these into one consistent, concise understanding "
        "(2-3 sentences) of the key visual changes."
    )

def _reason_synthesis(edit_instruction, reasoning_traces):
    traces_text = "\n\n".join(
        [f"Perspective {i+1}: {trace}" for i, trace in enumerate(reasoning_traces)]
    )
    return (
        f"Given this edit instruction: {edit_instruction}\n\n"
        f"Multiple perspectives on how the video would change:\n{traces_text}\n\n"
        "Synthesize these perspectives into a single, consistent understanding of the key changes:"
    )


PROMPT_STYLES = {
    "keyword": {
        "reference": _kw_reference,
        "edit": _kw_edit,
        "edit_reasoning": _kw_edit_reasoning,
        "edit_description": _kw_edit_description,
        "consistency_synthesis": _default_synthesis,
    },
    "dense": {
        "reference": _dense_reference,
        "edit": _dense_edit,
        "edit_reasoning": _dense_edit_reasoning,
        "edit_description": _dense_edit_description,
        "consistency_synthesis": _default_synthesis,
    },
    "reasoning": {
        "reference": _reason_reference,
        "edit": _reason_edit,
        "edit_reasoning": _reason_edit_reasoning,
        "edit_description": _reason_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
}


def get_prompts(style: str) -> dict:
    return PROMPT_STYLES[style]
