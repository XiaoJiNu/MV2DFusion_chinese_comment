import datetime
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner.hooks import HOOKS
from mmcv.runner import TextLoggerHook  # 正确导入
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class CustomTextLoggerHook(TextLoggerHook):
    """自定义文本日志钩子，使Loss输出格式更整齐."""
    
    def _log_info(self, log_dict, runner):
        # 提取基本训练信息
        mode = runner.mode
        epoch = log_dict.pop('epoch', 0)
        iter_num = log_dict.pop('iter', 0)
        total_iters = runner.max_iters
        
        # 处理学习率 - 修复此处的错误，兼容lr是浮点数或字典的情况
        lr = log_dict.pop('lr', None)
        if isinstance(lr, dict):
            # 如果是字典，使用原来的方式处理
            lr_str = ', '.join([f'{name}: {value:.3e}' for name, value in lr.items()])
        elif isinstance(lr, float):
            # 如果是浮点数，直接格式化
            lr_str = f'{lr:.3e}'
        else:
            # 如果是None或其他类型
            lr_str = 'None'
        
        eta_seconds = log_dict.pop('time', 0) * (runner.max_iters - runner.iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # 提取时间和内存信息
        time_str = f"{log_dict.pop('time', 0):.3f}"
        data_time_str = f"{log_dict.pop('data_time', 0):.3f}"
        memory_str = f"{log_dict.pop('memory', 0):.0f}"
        
        # 输出基本信息行
        log_str = f"Iter [{iter_num}/{total_iters}]  "
        log_str += f"lr: {lr_str}, eta: {eta_string}, "
        log_str += f"time: {time_str}, data_time: {data_time_str}, memory: {memory_str}"
        runner.logger.info(log_str)
        
        # 组织loss输出，每行最多3个指标
        loss_items = []
        for k, v in log_dict.items():
            if isinstance(v, float):
                loss_items.append(f"{k}: {v:.4f}")
            else:
                loss_items.append(f"{k}: {v}")
        
        # 每行最多3个指标，保持对齐
        chunk_size = 3
        for i in range(0, len(loss_items), chunk_size):
            chunk = loss_items[i:i+chunk_size]
            # 修改这里：确保每个指标固定宽度，并在每个指标之间添加空格
            # 使用更大的宽度值来确保足够的空间
            formatted_chunk = []
            for item in chunk:
                # 确保每个项目至少有35个字符宽度，使用空格填充
                formatted_chunk.append(f"{item:<35}")
            
            # 使用空格连接而不是直接拼接
            runner.logger.info("    " + "  ".join(formatted_chunk))