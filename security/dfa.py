
from __future__ import annotations

import logging
import os
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

class DFASensitiveFilter:

    def __init__(self, words_file: str = "./security/sensitive_words.txt"):
        self.words_file = words_file
        self._words: Set[str] = set()
        self._automaton = None
        self._load()

    def _load(self):
        path = self.words_file
        if not os.path.exists(path):
            logger.warning("DFA 词表不存在: %s，使用内置默认词表", path)
            words = [
                "身份证号", "银行卡号", "家庭住址", "电话号码",
                "自杀", "自残", "安乐死处方", "管制药品配方",
            ]
        else:
            with open(path, "r", encoding="utf-8") as f:
                words = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        self._words = set(words)

        try:
            import ahocorasick
            automaton = ahocorasick.Automaton()
            for w in self._words:
                automaton.add_word(w, w)
            automaton.make_automaton()
            self._automaton = automaton
            self._engine = "ahocorasick"
        except ImportError:
            self._automaton = None
            self._engine = "regex_fallback"
        logger.info("DFA filter loaded: %d words (engine=%s)", len(self._words), self._engine)

    def scan(self, text: str) -> List[str]:
        if not text:
            return []
        hits: List[str] = []
        if self._automaton is not None:
            for _end, word in self._automaton.iter(text):
                if word not in hits:
                    hits.append(word)
        else:
            for w in self._words:
                if w and w in text and w not in hits:
                    hits.append(w)
        return hits

    def contains_sensitive(self, text: str) -> bool:
        return bool(self.scan(text))
