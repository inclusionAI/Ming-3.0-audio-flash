# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/10/16
import re
import os
from .talker_pre_processor import TalkerPreProcessor
from .normalizer import Normalizer
from .talker_re import TalkerRE


class TalkerTN:
    def __init__(self, assets_dir=None):
        self.talker_pre_processor = TalkerPreProcessor()
        self.talker_re = TalkerRE()
        file_dir = assets_dir or os.path.dirname(os.path.realpath(__file__))
        default_re_cfg = f'{file_dir}/talker_re.json'
        if os.path.exists(default_re_cfg):
            self.talker_re.update(default_re_cfg)
        self.re_contains_chinese = re.compile(r'[\u4e00-\u9fff]')
        self.tn_zh = Normalizer(f'{file_dir}/zh_tn', ordertype='tn')
        self.tn_en = Normalizer(f'{file_dir}/en_tn', ordertype='en_tn')
        
    def __call__(self, text):
        text = self.talker_pre_processor(text)
        if len(text) == 0:
            return text
        
        is_chinese = bool(self.re_contains_chinese.search(text))
        if is_chinese:
            bak_text = text
            try:
                text = self.talker_re(text)
            except Exception:
                text = bak_text
            return self.tn_zh.normalize(text)
        return self.tn_en.normalize(text)
        
    def normalize(self, text):
        return self(text)
