# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import yaml
from boxmot.utils import BOXMOT, TRACKER_CONFIGS

def get_tracker_config(tracker_type):
    """Returns the path to the tracker configuration file."""
    return TRACKER_CONFIGS / f'{tracker_type}.yaml'

def create_tracker(tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None):
    """
    Creates and returns an instance of the specified tracker type.
    创建并返回指定跟踪链接类型的实例
    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision. 指示是否使用半精度的布尔值。
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker. 改进追踪器
    
    Returns:
    - An instance of the selected tracker.
    """
    
    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details['default'] for param, details in yaml_config.items()}
    else:
        tracker_args = evolve_param_dict

    # Arguments specific to ReID models 用于 ReID 模型的参数
    reid_args = {
        'reid_weights': reid_weights,
        'device': device,
        'half': half,
    }

    # Map tracker types to their corresponding classes
    tracker_mapping = {
        'strongsort': 'boxmot.trackers.strongsort.strongsort.StrongSort',
        'ocsort': 'boxmot.trackers.ocsort.ocsort.OcSort',
        'bytetrack': 'boxmot.trackers.bytetrack.bytetrack.ByteTrack',
        'botsort': 'boxmot.trackers.botsort.botsort.BotSort',
        'deepocsort': 'boxmot.trackers.deepocsort.deepocsort.DeepOcSort',
        'hybridsort': 'boxmot.trackers.hybridsort.hybridsort.HybridSort',
        'imprassoc': 'boxmot.trackers.imprassoc.imprassoctrack.ImprAssocTrack'
    }

    # Check if the tracker type exists in the mapping
    if tracker_type not in tracker_mapping:
        print('Error: No such tracker found.')
        exit()

    # Dynamically import and instantiate the correct tracker class 动态导入并实例化正确的跟踪器类
    module_path, class_name = tracker_mapping[tracker_type].rsplit('.', 1)
    tracker_class = getattr(__import__(module_path, fromlist=[class_name]), class_name)
    
    # For specific trackers, update tracker arguments with ReID parameters 对于特定跟踪链接，使用 ReID 参数更新跟踪链接参数
    if tracker_type in ['strongsort', 'botsort', 'deepocsort', 'hybridsort', 'imprassoc']:
        tracker_args['per_class'] = per_class
        tracker_args.update(reid_args)
        if tracker_type == 'strongsort':
            tracker_args.pop('per_class')  # per class not supported by
    else:
        tracker_args['per_class'] = per_class

    # Return the instantiated tracker class with arguments 返回带参数的实例化 tracker 类
    return tracker_class(**tracker_args)