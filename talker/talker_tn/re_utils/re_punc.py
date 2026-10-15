# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/10/21
from .re_base import REBase


class REPunc(REBase):
    name = 'punc'
    
    def __init__(self, txt):
        super().__init__(txt)

    def to_string(self):
        return self.txt
