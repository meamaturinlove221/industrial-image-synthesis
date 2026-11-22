from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manitwin_image_platform.texture_tasks.planner import build_group_plan
from manitwin_image_platform.texture_tasks.preview import paired_group_sheet,same_class_across_scenes_sheet

def main()->int:
    """导出指定配方中的成对与跨场景预览。"""
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--recipe',choices=('texture','layout'),default='layout')
    p.add_argument('--group-index',type=int,default=2); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    group=build_group_plan('train',args.group_index+1,8,args.seed,recipe=args.recipe)[args.group_index]
    paired_group_sheet(group,args.image_size,args.image_size).save(out/'同组AB严格对照.png')
    same_class_across_scenes_sheet(group,'A',args.image_size,args.image_size).save(out/'A类跨形状背景.png')
    same_class_across_scenes_sheet(group,'B',args.image_size,args.image_size).save(out/'B类跨形状背景.png')
    print(out); return 0
if __name__=='__main__': raise SystemExit(main())
