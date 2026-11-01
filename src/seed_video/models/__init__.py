"""Model backend registry."""


def model_spec(model_id):
    if model_id == "OpenGVLab/InternVL3_5-8B-Instruct":
        return ("InternVLChatModel", 28)
    from .qwen import model_spec as qwen_model_spec

    return qwen_model_spec(model_id)
