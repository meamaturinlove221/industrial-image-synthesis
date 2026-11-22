from __future__ import annotations
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class UIContractTests(unittest.TestCase):
    def test_major_actions_are_on_main_window_without_menu(self):
        text=(ROOT/'ui/texture_task_layout.py').read_text(encoding='utf-8')
        self.assertNotIn('tk.Menu',text)
        self.assertNotIn('Menu(',text)
        for label in ('刷新任务计划','生成当前组预览','生成正式任务集','审计现有任务集','打开输出根目录'):
            self.assertIn(label,text)
        self.assertIn('严格 A/B 同场景对照',text)
        self.assertIn('同一类跨形状/背景/颜色',text)

if __name__=='__main__':unittest.main()
