import getpass, os

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
