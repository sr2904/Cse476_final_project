from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from project import AnswerAgent

testfile = Path("cse_476_final_project_test_data.json")
outfile = Path("cse_476_final_project_answers.json")
examplefile = Path("cse476_final_project_dev_data.json")


def readtest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    return data


def readexamples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    return data


examples = readexamples(examplefile)
runner = AnswerAgent(examples=examples)


def run(tests: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    for item in tests:
        q = {
            "input": item.get("input", ""),
            "domain": item.get("domain", "unknown"),
        }
        ans = runner.solve(q)
        if not isinstance(ans, str):
            ans = str(ans)
        ans = ans.strip().replace("\n", " ")
        outputs.append({"output": ans})
    return outputs
