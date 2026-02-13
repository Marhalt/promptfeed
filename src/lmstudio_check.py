import importlib.util
import requests

def get_lm_studio_model_info():
    """
    Detect LM Studio installation and running state,
    then retrieve the current model name (displayName)
    and context length.
    Returns (model_name, context_length) or None.
    """
    # 1. Check if lmstudio library is installed
    if importlib.util.find_spec("lmstudio") is None:
        print("⚠️ The 'lmstudio' library is not installed. Install it with:\n   pip install lmstudio")
        return None

    import lmstudio as lms

    # 2. Check if LM Studio server is running
    base_url = "http://127.0.0.1:1234/v1"
    try:
        resp = requests.get(f"{base_url}/models", timeout=2)
        if resp.status_code != 200:
            print(f"⚠️ LM Studio API returned HTTP {resp.status_code}.")
            return None
    except requests.exceptions.RequestException:
        print(f"❌ Could not reach LM Studio at {base_url}. Is it running?")
        return None

    # 3. Retrieve model info
    try:
        model = lms.llm()
        context_length = model.get_context_length()

        # get_info() returns a dataclass-like object or dict
        info = model.get_info()
        if hasattr(info, "to_dict"):
            info = info.to_dict()

        # Try best fields for display name
        model_name = (
            info.get("displayName")
            or info.get("identifier")
            or info.get("modelKey")
            or "Unknown model"
        )

        print(f"  LM Studio running model: {model_name}")
        print(f"   Context length: {context_length} tokens")

        return model_name, context_length

    except Exception as e:
        print(f"⚠️ Could not retrieve model info from LM Studio: {e}")
        return None


# Example usage
if __name__ == "__main__":
    info = get_lmstudio_model_info()
    if info:
        model_name, ctx = info
        print(f"Using model '{model_name}' (context window = {ctx} tokens)")