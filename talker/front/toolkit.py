import re
from typing import Iterable

TOKENIZE_PATTERN = r"(?:[a-zA-Z]\.)+|[a-zA-Z]+(?:['\-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]|\s+|\S"

def tokenize_mixed_text(text: str):
    """
    将混合文本字符串处理为标记列表，并保留原始的空格。
    """
    return re.findall(TOKENIZE_PATTERN, text)

def tokenize_mixed_text_iterator(text_iterator: Iterable[str]):
    """
    Args:
        text_iterator (Iterable[str]): 一个产出字符串的迭代器。
            例如，一个打开的文件对象，或一个生成器表达式。
    Yields:
        Iterator[str]: 逐一产出文本中的标记（单词、中文、标点、空格等）。
    """
    # 函数本身就是一个生成器，将从 re.finditer 返回的迭代器中逐一产出结果
    for chunk in text_iterator:
        # re.finditer 返回一个迭代器，而不是列表
        for match in re.finditer(TOKENIZE_PATTERN, chunk):
            yield match.group(0)
