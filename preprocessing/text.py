import re
from typing import List


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

