"""关系任务数据集编排；完成状态只在深度像素审计通过后写入。"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from .audit import audit_texture_task_dataset
from .dataset_io import render_group, write_group_summary
from .planner import build_group_plan, load_group_plan, save_group_plan

RENDERER_VERSION = 'relation-tasks-v4-stable-structure'


@dataclass(frozen=True, slots=True)
class TextureTaskDatasetOptions:
    """正式任务集的独立配置，T与L只改变数据配方。"""
    output_root: str | Path
    dataset_name: str = 'relation_tasks_layout_v4'
    train_group_count: int = 3200
    val_group_count: int = 640
    images_per_class: int = 8
    image_size: int = 224
    seed: int = 42
    hard_pair_ratio: float = 0.25
    val_heldout_texture_ratio: float = 0.25
    val_heldout_shape_ratio: float = 0.25
    workers: int = 4
    recipe: str = 'layout'

    def configuration(self) -> dict:
        """直接比较影响数据内容的配置，不新增哈希。"""
        values = asdict(self)
        values['output_root'] = str(Path(self.output_root).resolve())
        values.pop('workers')
        return values


def _write(path: Path, payload: dict) -> None:
    """写入可直接检查的运行状态或摘要。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _prepare(root: Path, options: TextureTaskDatasetOptions):
    """新建冻结计划或读取配置一致的可续跑计划。"""
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'build_state.json'
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('renderer_version') != RENDERER_VERSION or state.get('options') != options.configuration():
            raise RuntimeError('现有任务集的生成版本或配置不同，请使用新的数据集名称。')
        return load_group_plan(root / 'train_plan.jsonl'), load_group_plan(root / 'val_plan.jsonl')
    if any(root.iterdir()):
        raise FileExistsError(f'输出目录非空且不是可续跑任务集：{root}')
    common = dict(images_per_class=options.images_per_class, hard_pair_ratio=options.hard_pair_ratio, recipe=options.recipe)
    train = build_group_plan('train', options.train_group_count, seed=options.seed,
                             heldout_texture_ratio=0, heldout_shape_ratio=0, **common)
    val = build_group_plan('val', options.val_group_count, seed=options.seed + 1,
                           heldout_texture_ratio=options.val_heldout_texture_ratio,
                           heldout_shape_ratio=options.val_heldout_shape_ratio, **common)
    save_group_plan(train, root / 'train_plan.jsonl')
    save_group_plan(val, root / 'val_plan.jsonl')
    _write(state_path, {
        'schema_version': 4, 'status': 'IN_PROGRESS', 'declared_stage': 'planned',
        'renderer_version': RENDERER_VERSION, 'recipe': options.recipe,
        'options': options.configuration(), 'completed_groups': 0,
    })
    return train, val


def _summary(root: Path, options: TextureTaskDatasetOptions, train, val) -> dict:
    """区分数据产物与尚未证明的模型训练效果。"""
    return {
        'schema_version': 4, 'renderer_version': RENDERER_VERSION,
        'dataset_name': options.dataset_name, 'dataset_root': str(root),
        'task_family': 'texture_relation', 'recipe': options.recipe,
        'declared_stage': 'generated', 'train_groups': len(train), 'val_groups': len(val),
        'images_per_class_per_group': options.images_per_class,
        'pairs_per_group': options.images_per_class, 'images_per_group': options.images_per_class * 2,
        'support_pairs': 4, 'train_images': len(train) * options.images_per_class * 2,
        'val_images': len(val) * options.images_per_class * 2,
        'image_size': options.image_size, 'seed_train': options.seed, 'seed_val': options.seed + 1,
        'task_kind_counts': {name: dict(Counter(group.task_kind for group in plan))
                             for name, plan in (('train', train), ('val', val))},
        'decision_factor_counts': {name: dict(Counter(group.decision_factor for group in plan))
                                   for name, plan in (('train', train), ('val', val))},
        'layout_domain_counts': {name: dict(Counter(group.layout_domain for group in plan if group.task_kind == 'layout'))
                                 for name, plan in (('train', train), ('val', val))},
        'val_heldout_texture_ratio': options.val_heldout_texture_ratio,
        'val_heldout_shape_ratio': options.val_heldout_shape_ratio,
        'design_contract': {
            'A_B_are_roles_not_global_labels': True,
            'same_scene_for_A_B_at_each_slot': True,
            'stable_prototype_structure': True, 'matched_visible_foreground_histogram': True,
            'not_bound_to_difficult_labels': True, 'model_architecture_change_required': False,
        },
    }


def build_texture_task_dataset(options: TextureTaskDatasetOptions, *,
                               progress_callback: Callable[[int, int, str], None] | None = None) -> dict:
    """生成、恢复并深度审计任务；任何实际失败终止本次生成。"""
    if options.images_per_class < 8 or options.image_size < 64 or not 1 <= options.workers <= 16:
        raise ValueError('每类至少8张，图像尺寸至少64，并行线程为1~16')
    if options.recipe not in {'texture', 'layout'}:
        raise ValueError('recipe必须为texture或layout')
    if options.hard_pair_ratio != 0.25:
        raise ValueError('正式T/L配方固定使用25%同家族困难组；请保持hard_pair_ratio=0.25。')
    root = Path(options.output_root).resolve() / options.dataset_name
    train, val = _prepare(root, options)
    state_path = root / 'build_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state.update(status='IN_PROGRESS', declared_stage='generating')
    _write(state_path, state)
    groups = train + val
    if progress_callback:
        progress_callback(0, len(groups), '逐组读取并验证续跑数据')
    try:
        with ThreadPoolExecutor(max_workers=options.workers, thread_name_prefix='relation-task') as pool:
            results = pool.map(lambda group: render_group(root, group, options.image_size), groups)
            for done, (group_id, _rendered) in enumerate(results, start=1):
                if done % 10 == 0 or done == len(groups):
                    if progress_callback:
                        progress_callback(done, len(groups), group_id)
                if done % 50 == 0:
                    state['completed_groups'] = done
                    _write(state_path, state)
        write_group_summary(root, train, 'train_groups.jsonl')
        write_group_summary(root, val, 'val_groups.jsonl')
        summary = _summary(root, options, train, val)
        _write(root / 'dataset_summary.json', summary)
        _write(root / 'sct_compatibility.json', {
            'schema_version': 4, 'train_root': str(root / 'train'), 'val_root': str(root / 'val'),
            'group_layout': '<root>/<split>/texture_relation/group_xxxxxx/A|B/000.png...',
            'support_pairs': 4, 'pairs_per_group': options.images_per_class,
            'model_architecture_change_required': False,
            'note_cn': '训练读取group.json即可，不导入本平台；训练和Difficult结果需另行验证。',
        })
        report = audit_texture_task_dataset(root, deep_image_check=True)
        _write(root / 'dataset_audit.json', report)
        if not report['passed']:
            raise RuntimeError(f'深度审计失败：{report["errors"][:5]}')
        summary['declared_stage'] = 'audited'
        _write(root / 'dataset_summary.json', summary)
        state.update(status='COMPLETED', declared_stage='audited', completed_groups=len(groups), audit_passed=True)
        _write(state_path, state)
        if progress_callback:
            progress_callback(len(groups), len(groups), '完成并通过深度像素审计')
        return summary
    except Exception as exc:
        state.update(status='FAILED', declared_stage='failed', error=str(exc), audit_passed=False)
        _write(state_path, state)
        raise
