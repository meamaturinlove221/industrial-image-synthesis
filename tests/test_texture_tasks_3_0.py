from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import ImageChops, ImageStat

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

from manitwin_image_platform.texture_tasks.prototype import (
    ALL_TEXTURE_FAMILIES, HELDOUT_TEXTURE_FAMILIES, TRAIN_TEXTURE_FAMILIES,
    make_distinct_same_family_pair, prototype_distance, sample_texture_prototype,
)
from manitwin_image_platform.texture_tasks.shape_bank import TRAIN_SHAPE_FAMILIES,HELDOUT_SHAPE_FAMILIES,make_shape_spec,render_shape_mask
from manitwin_image_platform.texture_tasks.scene import build_group_scenes
from manitwin_image_platform.texture_tasks.planner import build_group_plan
from manitwin_image_platform.texture_tasks.renderer import render_texture_scene,texture_map
from manitwin_image_platform.texture_tasks.dataset import TextureTaskDatasetOptions,build_texture_task_dataset
from manitwin_image_platform.texture_tasks.audit import audit_texture_task_dataset
from manitwin_image_platform.texture_tasks.preview import paired_group_sheet,same_class_across_scenes_sheet

class TextureTaskTests(unittest.TestCase):
    def test_all_24_texture_families_render(self):
        scene=build_group_scenes('train',0,8,42)[0]
        signatures=set()
        for idx,family in enumerate(ALL_TEXTURE_FAMILIES):
            proto=sample_texture_prototype(f'p{idx}',family,1000+idx)
            image,_=render_texture_scene(proto,scene,128,128,role='A')
            self.assertEqual(image.size,(128,128))
            self.assertGreater(sum(ImageStat.Stat(image).stddev),5)
            signatures.add(tuple(round(x,2) for x in ImageStat.Stat(image.resize((16,16))).mean))
        self.assertGreaterEqual(len(signatures),8)


    def test_all_20_shape_families_render_nonempty_and_bounded(self):
        all_shapes=TRAIN_SHAPE_FAMILIES+HELDOUT_SHAPE_FAMILIES
        signatures=set()
        for idx,family in enumerate(all_shapes):
            mask=render_shape_mask(make_shape_spec(family,1000+idx),256,180)
            ratio=ImageStat.Stat(mask).mean[0]/255.0
            self.assertGreater(ratio,0.20,family)
            self.assertLess(ratio,0.80,family)
            self.assertIsNotNone(mask.getbbox(),family)
            signatures.add(mask.resize((32,24)).tobytes())
        self.assertEqual(len(signatures),len(all_shapes))

    def test_same_family_hard_pair_has_min_distance(self):
        a,b=make_distinct_same_family_pair('g','horizontal_bands',1,2,min_distance=.22)
        self.assertEqual(a.family,b.family)
        self.assertGreaterEqual(prototype_distance(a,b),.22)

    def test_group_plan_uses_roles_and_heldout_only_in_val(self):
        train=build_group_plan('train',80,8,42,hard_pair_ratio=.25,heldout_texture_ratio=0,heldout_shape_ratio=0)
        val=build_group_plan('val',40,8,43,hard_pair_ratio=.25,heldout_texture_ratio=.25,heldout_shape_ratio=.25)
        self.assertEqual(len(train),80);self.assertEqual(len(val),40)
        self.assertFalse(any(g.prototype_a.family in HELDOUT_TEXTURE_FAMILIES or g.prototype_b.family in HELDOUT_TEXTURE_FAMILIES for g in train))
        self.assertGreater(sum(g.texture_domain=='heldout' for g in val),0)
        self.assertGreater(sum(g.hard_pair for g in val),0)
        self.assertFalse(any(s.shape_family in HELDOUT_SHAPE_FAMILIES for g in train for s in g.scenes))
        self.assertGreater(sum(s.shape_family in HELDOUT_SHAPE_FAMILIES for g in val for s in g.scenes),0)
        # Roles should not pin one family permanently to A.
        a_fams={g.prototype_a.family for g in train};b_fams={g.prototype_b.family for g in train}
        self.assertGreaterEqual(len(a_fams),15);self.assertGreaterEqual(len(b_fams),15)


    def test_full_default_plan_is_balanced_and_role_symmetric(self):
        """新正式规模保留家族平衡，未见纹理不固定在一个角色。"""
        train=build_group_plan('train',3200,8,42,recipe='texture',heldout_texture_ratio=0,heldout_shape_ratio=0)
        val=build_group_plan('val',640,8,43,recipe='texture',heldout_texture_ratio=.25,heldout_shape_ratio=.25)
        self.assertEqual(sum(g.hard_pair for g in train),800)
        self.assertEqual(sum(g.hard_pair for g in val),160)
        self.assertEqual(sum(g.texture_domain=='heldout' for g in val),160)
        for plan,limit in ((train,3),(val,3)):
            a=Counter(g.prototype_a.family for g in plan); b=Counter(g.prototype_b.family for g in plan)
            totals=[a[f]+b[f] for f in TRAIN_TEXTURE_FAMILIES]
            self.assertLessEqual(max(totals)-min(totals),limit)
        held_a=sum(g.prototype_a.family in HELDOUT_TEXTURE_FAMILIES for g in val)
        held_b=sum(g.prototype_b.family in HELDOUT_TEXTURE_FAMILIES for g in val)
        self.assertEqual(held_a,160); self.assertEqual(held_b,160)

    def test_default_group_has_eight_distinct_backgrounds_colors_and_angles(self):
        group=build_group_plan('train',1,8,42)[0]
        self.assertEqual(len({s.background_style for s in group.scenes}),8)
        self.assertEqual(len({s.shape_family for s in group.scenes}),8)
        # 分层颜色与角度应确保类别内8张具有明显干扰跨度。
        hues=sorted(s.base_h for s in group.scenes)
        self.assertGreater(max(hues)-min(hues),0.70)
        angles=[s.angle_deg for s in group.scenes]
        self.assertGreater(max(angles)-min(angles),130.0)

    def test_one_group_has_8_distinct_scenes_and_AB_share_nuisance(self):
        group=build_group_plan('train',1,8,42)[0]
        self.assertEqual(len(group.scenes),8)
        self.assertEqual(len({s.scene_id for s in group.scenes}),8)
        self.assertEqual(len({s.shape_family for s in group.scenes}),8)
        for scene in group.scenes:
            a,ma=render_texture_scene(group.prototype_a,scene,160,160,role='A')
            b,mb=render_texture_scene(group.prototype_b,scene,160,160,role='B')
            for key in ('scene_id','shape_family','shape_domain','background_style','angle_deg'):
                self.assertEqual(ma[key],mb[key])
            # Background corner is pixel-identical because scene is shared.
            self.assertEqual(a.getpixel((2,2)),b.getpixel((2,2)))

    def test_texture_is_object_local_not_frame_fixed(self):
        # Horizontal prototype should rotate with the object because texture is applied before object rotation.
        from dataclasses import replace
        group=build_group_plan('train',1,8,42)[0]
        proto=sample_texture_prototype('h','horizontal_bands',123)
        scene0=replace(group.scenes[0],angle_deg=0.0,base_h=.1,accent_h=.6)
        scene90=replace(group.scenes[0],angle_deg=90.0,base_h=.1,accent_h=.6)
        a,_=render_texture_scene(proto,scene0,180,180,role='A'); b,_=render_texture_scene(proto,scene90,180,180,role='A')
        self.assertGreater(sum(ImageStat.Stat(ImageChops.difference(a,b)).mean),15)

    def test_preview_sheets(self):
        group=build_group_plan('train',1,8,42)[0]
        pair=paired_group_sheet(group,128,128,thumb=90); cls=same_class_across_scenes_sheet(group,'A',128,128,thumb=90)
        self.assertGreater(pair.height,200);self.assertGreater(pair.width,700)
        self.assertGreater(cls.width,300);self.assertGreater(cls.height,150)

    def test_tiny_dataset_build_audit_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            opts=TextureTaskDatasetOptions(output_root=td,dataset_name='tiny',train_group_count=10,val_group_count=4,images_per_class=8,image_size=96,seed=7,workers=2,val_heldout_texture_ratio=.25,val_heldout_shape_ratio=.25)
            summary=build_texture_task_dataset(opts)
            root=Path(summary['dataset_root'])
            self.assertEqual(summary['train_images'],160);self.assertEqual(summary['val_images'],64)
            report=audit_texture_task_dataset(root,deep_image_check=True)
            self.assertTrue(report['passed'],report.get('errors'))
            self.assertEqual(report['train_val_prototype_seed_overlap'],0)
            self.assertEqual(report['duplicate_prototype_pairs'],0)
            self.assertTrue((root/'sct_compatibility.json').is_file())
            # Corrupt completeness by deleting one image, then resume.
            victim=root/'train/texture_relation/group_000000/A/000.png'; victim.unlink()
            self.assertFalse(audit_texture_task_dataset(root)['passed'])
            build_texture_task_dataset(opts)
            self.assertTrue(audit_texture_task_dataset(root,deep_image_check=True)['passed'])
            self.assertTrue(victim.is_file())
            # 只保留文件名但截断为0B也必须被续跑识别并重建。
            victim.write_bytes(b'')
            build_texture_task_dataset(opts)
            self.assertGreater(victim.stat().st_size,128)
            self.assertTrue(audit_texture_task_dataset(root,deep_image_check=True)['passed'])

    def test_reject_mixed_config_resume(self):
        with tempfile.TemporaryDirectory() as td:
            a=TextureTaskDatasetOptions(output_root=td,dataset_name='x',train_group_count=2,val_group_count=1,images_per_class=8,image_size=80,seed=1,workers=1)
            build_texture_task_dataset(a)
            b=TextureTaskDatasetOptions(output_root=td,dataset_name='x',train_group_count=3,val_group_count=1,images_per_class=8,image_size=80,seed=1,workers=1)
            with self.assertRaises(RuntimeError):build_texture_task_dataset(b)

if __name__=='__main__':unittest.main()
