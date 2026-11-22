"""共享场景的二维渲染与最终可见前景直方图匹配。"""
import math
import numpy as np
from PIL import Image, ImageFilter
from .layout import LayoutPrototype, layout_map
from .prototype import TexturePrototype
from .scene import SceneSpec
from .shape_bank import make_shape_spec, render_shape_mask
from .texture_map import _hsv, _noise, _normalize, texture_map


def _background(scene: SceneSpec, width: int, height: int) -> Image.Image:
    """只使用场景种子生成A/B共享背景。"""
    rng = np.random.default_rng(scene.background_seed)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    style = scene.background_style
    if style == 'soft_gradient':
        theta = float(rng.uniform(0, math.pi)); field = math.cos(theta)*x + math.sin(theta)*y
    elif style == 'radial_gradient':
        cx,cy=float(rng.uniform(-.4,.4)),float(rng.uniform(-.4,.4)); field=np.sqrt((x-cx)**2+(y-cy)**2)
    elif style == 'cloud_noise': field=.8*_noise(rng,width,height,24)+.2*_noise(rng,width,height,10)
    elif style == 'coarse_noise': field=_noise(rng,width,height,6)
    elif style == 'blurred_blocks': field=_noise(rng,width,height,30)
    elif style == 'diagonal_fiber': field=.55*np.sin((x+y)*math.pi*float(rng.uniform(2,5)))+.45*_noise(rng,width,height,8)
    elif style == 'low_contrast_cells': field=np.cos(x*math.pi*float(rng.uniform(2,4)))*np.cos(y*math.pi*float(rng.uniform(2,4)))
    elif style == 'industrial_mottle': field=.55*_noise(rng,width,height,14)+.45*_noise(rng,width,height,5)
    else: raise ValueError(f'未知背景：{style}')
    field=_normalize(field)
    h=float(rng.random()); s=float(rng.uniform(.04,.36)); v=float(rng.uniform(.20,.88))
    c1=_hsv(h,s,v); c2=_hsv((h+float(rng.uniform(-.16,.16)))%1.0,min(.52,s+.12),min(.96,max(.12,v+float(rng.uniform(-.22,.22)))))
    rgb=c1[None,None,:]*(1-field[:,:,None])+c2[None,None,:]*field[:,:,None]
    return Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8),'RGB').filter(ImageFilter.GaussianBlur(radius=.6))


def _transform_object(obj: Image.Image, scene: SceneSpec, width: int, height: int) -> tuple[Image.Image, int, int]:
    """对纹理、布局和遮罩统一应用缩放、模糊和旋转。"""
    shape = make_shape_spec(scene.shape_family, scene.shape_seed)
    if scene.object_blur > 0.05:
        obj = obj.filter(ImageFilter.GaussianBlur(radius=scene.object_blur))
    canonical_h = max(32, int(height * scene.scale * 0.68))
    canonical_w = max(32, int(canonical_h * shape.aspect_ratio * scene.aspect_scale))
    factor = min(1.0, width * 0.84 / canonical_w, height * 0.78 / canonical_h)
    obj = obj.resize((max(24, int(canonical_w * factor)), max(24, int(canonical_h * factor))), Image.Resampling.LANCZOS)
    obj = obj.rotate(scene.angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    x, y = int(scene.center_x * width - obj.width / 2), int(scene.center_y * height - obj.height / 2)
    return obj, x, y


def render_foreground_mask(scene: SceneSpec, width: int, height: int) -> np.ndarray:
    """还原最终可见前景遮罩，供真实像素审计使用。"""
    shape = make_shape_spec(scene.shape_family, scene.shape_seed)
    obj = Image.new('RGBA', (256, 180), (0, 0, 0, 0))
    obj.putalpha(render_shape_mask(shape, 256, 180))
    obj, x, y = _transform_object(obj, scene, width, height)
    mask = Image.new('L', (width, height), 0)
    mask.paste(obj.getchannel('A'), (x, y))
    return np.asarray(mask) > 0


def render_texture_scene(proto: TexturePrototype | LayoutPrototype, scene: SceneSpec, width: int, height: int, *, role: str) -> tuple[Image.Image, dict]:
    """渲染一个原型；正式配对输出再统一可见前景直方图。"""
    instance_seed = proto.seed + int(scene.scene_id.rsplit('_', 1)[-1]) * 1009
    local_w,local_h=256,180
    shape=make_shape_spec(scene.shape_family,scene.shape_seed)
    mask=render_shape_mask(shape,local_w,local_h)
    tex = layout_map(proto, np.asarray(mask)) if isinstance(proto, LayoutPrototype) else texture_map(proto, local_w, local_h, instance_seed)

    base=_hsv(scene.base_h,scene.base_s,scene.base_v); accent=_hsv(scene.accent_h,scene.accent_s,scene.accent_v)
    # Fixed scene-level contrast range; prototype contrast only modulates spatial amplitude slightly.
    amp = 0.78
    mix=np.clip(0.5 + (tex[:,:,None]-0.5)*amp*2.0,0,1)
    rgb=base[None,None,:]*(1-mix)+accent[None,None,:]*mix
    xx=np.linspace(-1,1,local_w,dtype=np.float32)[None,:]; yy=np.linspace(-1,1,local_h,dtype=np.float32)[:,None]
    theta=math.radians(scene.light_angle_deg); light=_normalize(.5+.5*(math.cos(theta)*xx+math.sin(theta)*yy))
    rgb=np.clip(rgb*(.88+scene.light_strength*light[:,:,None]),0,1)
    obj=Image.fromarray((rgb*255).astype(np.uint8),'RGB').convert('RGBA'); obj.putalpha(mask)
    obj, x, y = _transform_object(obj, scene, width, height)
    bg=_background(scene,width,height).convert('RGBA')
    alpha=obj.getchannel('A'); sm=Image.new('L',(width,height),0); sm.paste(alpha,(x+4,y+5)); sm=sm.filter(ImageFilter.GaussianBlur(5.0))
    shadow=Image.new('RGBA',(width,height),(0,0,0,0)); shadow.putalpha(sm.point(lambda p:int(p*scene.shadow_strength))); bg=Image.alpha_composite(bg,shadow); bg.alpha_composite(obj,(x,y))
    meta={'role':role,'prototype_id':proto.prototype_id,'texture_family':proto.family,'texture_family_cn':proto.family_cn,'scene_id':scene.scene_id,'shape_family':scene.shape_family,'shape_family_cn':scene.shape_family_cn,'shape_domain':scene.shape_domain,'background_style':scene.background_style,'angle_deg':scene.angle_deg}
    return bg.convert('RGB'),meta


def pair_pixel_metrics(image_a: Image.Image, image_b: Image.Image, mask: np.ndarray) -> dict:
    """从最终RGB像素核对背景、面积、直方图与可见差异。"""
    a, b = np.asarray(image_a.convert('RGB')), np.asarray(image_b.convert('RGB'))
    histogram_error = max(int(np.abs(np.bincount(a[mask, c], minlength=256) - np.bincount(b[mask, c], minlength=256)).sum()) for c in range(3))
    return {
        'foreground_pixels': int(mask.sum()),
        'background_equal': bool(np.array_equal(a[~mask], b[~mask])),
        'foreground_histogram_l1': histogram_error,
        'foreground_rgb_mae': float(np.abs(a[mask].astype(float) - b[mask].astype(float)).mean()),
    }


def render_texture_pair(group, scene: SceneSpec, width: int, height: int) -> tuple[Image.Image, Image.Image, dict]:
    """保持空间排列，精确匹配两侧最终可见前景的RGB直方图。"""
    image_a, _ = render_texture_scene(group.prototype_a, scene, width, height, role='A')
    image_b, _ = render_texture_scene(group.prototype_b, scene, width, height, role='B')
    mask = render_foreground_mask(scene, width, height)
    a, b = np.asarray(image_a), np.array(image_b)
    values = b[mask].copy()
    for channel in range(3):
        order = np.argsort(values[:, channel], kind='stable')
        values[order, channel] = np.sort(a[mask, channel])
    b[mask] = values
    image_b = Image.fromarray(b, 'RGB')
    return image_a, image_b, pair_pixel_metrics(image_a, image_b, mask)
