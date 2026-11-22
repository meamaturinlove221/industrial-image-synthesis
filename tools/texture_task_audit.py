from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manitwin_image_platform.texture_tasks.audit import audit_texture_task_dataset

def main()->int:
    """默认执行深度像素审计，快速检查必须明确选择。"""
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root',required=True)
    p.add_argument('--deep',action='store_true',help='保留原深度审计命令；现在默认深检')
    p.add_argument('--quick',action='store_true',help='只检查结构，不能作为正式训练凭据')
    args=p.parse_args()
    report=audit_texture_task_dataset(args.dataset_root,deep_image_check=not args.quick)
    out=Path(args.dataset_root)/'dataset_audit.json'; out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report.get('passed') else 2
if __name__=='__main__': raise SystemExit(main())
