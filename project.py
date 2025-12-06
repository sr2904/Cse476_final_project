import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

apitoken = os.getenv("OPENAI_API_KEY", "cse476")
apiurl = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
modelname = os.getenv("MODEL_NAME", "bens_model")


def askmodel(
    prompt: str,
    system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
    model: str = modelname,
    temperature: float = 0.0,
    timeout: int = 60,
) -> Dict[str, Any]:
    url = f"{apiurl}/chat/completions"
    headers = {
        "Authorization": f"Bearer {apitoken}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "text": text,
                "raw": data,
                "status": status,
                "error": None,
                "headers": hdrs,
            }
        else:
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "status": status,
                "error": str(err_text),
                "headers": hdrs,
            }
    except requests.RequestException as e:
        return {
            "ok": False,
            "text": None,
            "raw": None,
            "status": -1,
            "error": str(e),
            "headers": {},
        }


