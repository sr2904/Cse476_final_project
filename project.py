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


class AnswerAgent:
    def _init_(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        model: str = modelname,
        limit: int = 20,
    ):
        self.model = model
        self.limit = limit
        self.examples: List[Dict[str, Any]] = examples or []
        self.domains = self.group(self.examples)

    def group(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in data:
            dom = item.get("domain", "unknown")
            out.setdefault(dom, []).append(item)
        return out

    def split(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def match(self, a: str, b: str) -> float:
        sa, sb = set(self.split(a)), set(self.split(b))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def pick(self, domain: str, question: str, k: int = 2) -> List[Dict[str, Any]]:
        base = self.domains.get(domain, self.examples)
        if not base:
            return []
        scored = [(self.match(question, ex.get("input", "")), ex) for ex in base]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored[:k] if score > 0]

    def prompt(self, domain: str, question: str, kshots: int = 2) -> str:
        shots = self.pick(domain, question, k=kshots)
        text = ""
        for ex in shots:
            q = str(ex.get("input", "")).strip()
            a = str(ex.get("output", "")).strip()
            text += f"Q: {q}\nA: {a}\n\n"
        out = (
            f"You are solving {domain} reasoning problems.\n"
            "Here are some solved examples.\n\n"
            f"{text}"
            "Now answer this new question.\n"
            f"Q: {question.strip()}\n\n"
            "Think if needed, but end with:\n"
            "FINAL_ANSWER: <answer>"
        )
        return out

    def extract(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text)
        if m:
            ans = m.group(1)
        else:
            ans = text
        ans = ans.strip().replace("\n", " ")
        ans = re.sub(r"\s+", " ", ans).strip()
        ans = re.sub(r"^FINAL_ANSWER\s*:\s*", "", ans, flags=re.IGNORECASE)
        return ans

    def tryanswers(
        self,
        domain: str,
        question: str,
        samples: int = 2,
        kshots: int = 2,
        temperature: float = 0.7,
    ) -> List[str]:
        out: List[str] = []
        maxsamples = max(1, self.limit - 2)
        n = max(1, min(samples, maxsamples))
        for _ in range(n):
            p = self.prompt(domain, question, kshots)
            r = askmodel(
                p,
                system="You are a careful solver. Include reasoning but end with FINAL_ANSWER.",
                model=self.model,
                temperature=temperature,
            )
            raw = (r.get("text") or "").strip()
            ans = self.extract(raw)
            out.append(ans)
            time.sleep(0.05)
        return out

    def decide(self, items: List[str]) -> str:
        arr = [x.strip() for x in items if x and x.strip()]
        if not arr:
            return ""
        count = Counter(arr)
        return count.most_common(1)[0][0]

    def review(self, domain: str, question: str, draft: str) -> str:
        if not draft:
            return draft
        p = (
            "You are checking if an answer is correct.\n\n"
            f"Domain: {domain}\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{draft}\n\n"
            "If it is correct, reply:\n"
            "ACCEPT: <same answer>\n\n"
            "If it is wrong or incomplete, reply:\n"
            "REVISE: <better answer>"
        )
        r = askmodel(
            p,
            system="Check the answer and either ACCEPT or REVISE it.",
            model=self.model,
            temperature=0.0,
        )
        text = (r.get("text") or "").strip()
        if text.startswith("REVISE:"):
            result = text[len("REVISE:") :].strip()
        elif text.startswith("ACCEPT:"):
            result = text[len("ACCEPT:") :].strip()
        else:
            result = draft
        result = result.replace("\n", " ")
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def final(self, question: str, draft: str) -> str:
        if not draft:
            return ""
        p = (
            "Given a question and a draft answer, return ONLY the final answer.\n"
            "Do not include any explanation.\n\n"
            f"Question:\n{question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Final answer only:"
        )
        r = askmodel(
            p,
            system="Return only the minimal final answer string.",
            model=self.model,
            temperature=0.0,
        )
        t = (r.get("text") or "").strip()
        t = t.replace("\n", " ")
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"^[A-D]\W+", "", t).strip()
        return t

    def solve(self, item: Dict[str, Any]) -> str:
        question = item.get("input", "")
        domain = item.get("domain", "unknown")
        dom = (domain or "").lower()
        samples = 3 if "math" in dom else 2
        tries = self.tryanswers(domain, question, samples=samples, kshots=2)
        base = self.decide(tries)
        checked = self.review(domain, question, base)
        out = self.final(question, checked)
        return out