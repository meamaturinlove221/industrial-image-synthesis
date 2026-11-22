from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manitwin_image_platform.texture_tasks.dataset import TextureTaskDatasetOptions, build_texture_task_dataset

def main()->int:
    """按固定T/L配方生成任务，并在实际像素审计后交付。"""
    p=argparse.ArgumentParser(description='生成通用纹理与布局关系任务集')
    p.add_argument('--output-root',required=True); p.add_argument('--dataset-name',default='relation_tasks_layout_v4')
    p.add_argument('--recipe',choices=('texture','layout'),default='layout')
    p.add_argument('--train-groups',type=int,default=3200); p.add_argument('--val-groups',type=int,default=640)
    p.add_argument('--images-per-class',type=int,default=8); p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--hard-pair-ratio',type=float,default=.25,help='T/L固定配方必须为0.25')
    p.add_argument('--val-heldout-texture-ratio',type=float,default=.25); p.add_argument('--val-heldout-shape-ratio',type=float,default=.25)
    p.add_argument('--workers',type=int,default=4); args=p.parse_args()
    def prog(done,total,stage):
        """把进度传给界面与命令行。"""
        print(f'@@TEXTURE_TASK_PROGRESS@@|{done}|{total}|{stage}',flush=True)
    result=build_texture_task_dataset(TextureTaskDatasetOptions(
        output_root=args.output_root,dataset_name=args.dataset_name,train_group_count=args.train_groups,val_group_count=args.val_groups,
        images_per_class=args.images_per_class,image_size=args.image_size,seed=args.seed,hard_pair_ratio=args.hard_pair_ratio,
        val_heldout_texture_ratio=args.val_heldout_texture_ratio,val_heldout_shape_ratio=args.val_heldout_shape_ratio,
        workers=args.workers,recipe=args.recipe),progress_callback=prog)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
