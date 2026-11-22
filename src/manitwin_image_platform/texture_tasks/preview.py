from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .planner import TextureTaskGroup
from .prototype import effective_parameters
from .renderer import render_texture_pair


def _font(size: int, bold: bool = False):
    """选择本机可用的中文字体。"""
    candidates = [
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def paired_group_sheet(group: TextureTaskGroup, width: int = 224, height: int = 224, thumb: int = 150, scene_columns: int = 4) -> Image.Image:
    """使用与正式输出相同的配对渲染显示控制变量。"""
    count = len(group.scenes)
    rows = (count + scene_columns - 1) // scene_columns
    cap_h = 38
    pair_w = thumb * 2 + 8
    pair_h = thumb + cap_h
    sheet = Image.new('RGB', (scene_columns * pair_w, rows * pair_h), 'white')
    draw = ImageDraw.Draw(sheet); font=_font(13); bold=_font(14, True)
    for idx, scene in enumerate(group.scenes):
        a,b,_=render_texture_pair(group,scene,width,height)
        a=a.resize((thumb,thumb),Image.Resampling.LANCZOS); b=b.resize((thumb,thumb),Image.Resampling.LANCZOS)
        r,c=divmod(idx,scene_columns); x=c*pair_w; y=r*pair_h
        sheet.paste(a,(x,y)); sheet.paste(b,(x+thumb+8,y))
        draw.text((x+4,y+4),'A',font=bold,fill='white',stroke_width=2,stroke_fill='black')
        draw.text((x+thumb+12,y+4),'B',font=bold,fill='white',stroke_width=2,stroke_fill='black')
        label=f'{idx:02d} {scene.shape_family_cn} / {scene.background_style_cn}'
        draw.text((x+3,y+thumb+3),label,font=font,fill='black')
    return sheet


def same_class_across_scenes_sheet(group: TextureTaskGroup, role: str, width: int = 224, height: int = 224, columns: int = 4, thumb: int = 150) -> Image.Image:
    """展示同类在不同形状和场景中的共同结构。"""
    rows=(len(group.scenes)+columns-1)//columns; cap_h=36
    sheet=Image.new('RGB',(columns*thumb,rows*(thumb+cap_h)),'white'); draw=ImageDraw.Draw(sheet); font=_font(14)
    for idx,scene in enumerate(group.scenes):
        a,b,_=render_texture_pair(group,scene,width,height)
        img = (a if role.upper() == 'A' else b).resize((thumb,thumb),Image.Resampling.LANCZOS)
        r,c=divmod(idx,columns); x=c*thumb; y=r*(thumb+cap_h); sheet.paste(img,(x,y)); draw.text((x+3,y+thumb+2),f'{scene.shape_family_cn}',font=font,fill='black')
    return sheet


def group_summary_text(group: TextureTaskGroup) -> str:
    """解释当前分类依据、实际差异和类内允许的变化。"""
    factors = {
        'texture_family': '纹理家族不同', 'texture_structure': '同家族纹理结构不同',
        'region_position': '区域位置', 'region_arrangement': '区域排列',
        'region_dispersion': '聚集与分散', 'region_connectivity': '连接与分离',
    }
    lines = [f'任务：{group.group_id}（{"训练" if group.split == "train" else "验证"}）',
             f'当前判别依据：{factors[group.decision_factor]}',
             f'配方：{"T 纹理对照" if group.recipe == "texture" else "L 纹理与布局关系"}',
             f'A：{group.prototype_a.family_cn}；B：{group.prototype_b.family_cn}']
    if group.hard_pair:
        names = {'frequency': '主方向频率', 'secondary_frequency': '次方向频率',
                 'thickness': '线宽', 'warp': '局部扭曲', 'structure': '结构混合',
                 'dropout': '缺失比例', 'softness': '纹理柔化'}
        lines.append('本组实际变化的有效属性（A / B）：')
        for name in effective_parameters(group.prototype_a.family):
            a, b = getattr(group.prototype_a, name), getattr(group.prototype_b, name)
            if abs(a - b) > 1e-6:
                lines.append(f'  {names[name]}：{a:.3f} / {b:.3f}')
    elif group.task_kind == 'layout':
        rules = {'region_position': ('左侧', '右侧'), 'region_arrangement': ('横向排列', '纵向排列'),
                 'region_dispersion': ('聚集', '分散'), 'region_connectivity': ('连接', '分离')}
        labels = rules[group.decision_factor]
        lines.append(f'物体局部规则：A {labels[group.prototype_a.variant]}；B {labels[group.prototype_b.variant]}')
        domain = '验证留出组合' if group.layout_domain == 'heldout' else '训练可见组合'
        lines.append(f'区域配置：{group.prototype_a.element_count} 个元素，{domain}；两侧可见色块面积相同。')
    lines += [f'每类 {len(group.scenes)} 张；同槽 A/B 共享同一场景和前景颜色直方图。',
              '类内保持原型结构；只变化形状、背景、配色、姿态、尺度、光照和轻微模糊。',
              '纹理实例仅加微弱噪声；A/B只是当前任务角色，生成预览不代表模型训练有效。']
    return '\n'.join(lines)
