#!/usr/bin/env python3
"""
模型量化类型分析脚本
用于分析 PyTorch 模型的量化格式
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch


def analyze_model(model_path: str) -> Dict[str, Any]:
    """分析模型权重，判断量化类型"""

    # 加载模型
    if model_path.endswith('.safetensors'):
        from safetensors import safe_open
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    # 分析结果
    result = {
        'total_tensors': len(state_dict),
        'dtype_counts': defaultdict(int),
        'quant_weights': [],       # qweight 类型的权重
        'weight_scales': [],       # 权重 scales
        'input_scales': [],        # 输入 scales (静态量化标志)
        'zero_points': [],         # zero points (非对称量化标志)
        'normal_weights': [],      # 普通权重
    }

    for name, tensor in state_dict.items():
        dtype_str = str(tensor.dtype)
        result['dtype_counts'][dtype_str] += 1

        name_lower = name.lower()

        # 检测量化权重
        if 'qweight' in name_lower or (name_lower.endswith('.weight') and tensor.dtype in [torch.int8, torch.int4, torch.uint8]):
            result['quant_weights'].append({
                'name': name,
                'dtype': dtype_str,
                'shape': list(tensor.shape)
            })
        # 检测输入 scales (静态量化) - 需要在权重 scales 之前检测
        elif 'input_scale' in name_lower or 'act_scale' in name_lower:
            result['input_scales'].append({
                'name': name,
                'dtype': dtype_str,
                'shape': list(tensor.shape)
            })
        # 检测权重 scales
        elif name_lower.endswith('.scales') or 'weight_scale' in name_lower:
            result['weight_scales'].append({
                'name': name,
                'dtype': dtype_str,
                'shape': list(tensor.shape)
            })
        # 检测 zero points
        elif 'zero' in name_lower or 'qzeros' in name_lower:
            result['zero_points'].append({
                'name': name,
                'dtype': dtype_str,
                'shape': list(tensor.shape)
            })
        # 普通权重
        elif name_lower.endswith('.weight'):
            result['normal_weights'].append({
                'name': name,
                'dtype': dtype_str,
                'shape': list(tensor.shape)
            })

    return result


def determine_quant_type(result: Dict[str, Any]) -> Dict[str, str]:
    """根据分析结果判断量化类型"""

    info = {
        'is_quantized': False,
        'weight_dtype': 'unknown',
        'quant_granularity': 'unknown',  # per-tensor, per-channel, per-group
        'quant_type': 'unknown',         # dynamic, static
        'symmetry': 'unknown',           # symmetric, asymmetric
        'summary': ''
    }

    # 判断是否量化
    if result['quant_weights']:
        info['is_quantized'] = True

        # 判断权重数据类型
        dtypes = set(w['dtype'] for w in result['quant_weights'])
        if 'torch.int8' in dtypes:
            info['weight_dtype'] = 'INT8'
        elif 'torch.int4' in dtypes or 'torch.uint8' in dtypes:
            info['weight_dtype'] = 'INT4'
        elif 'torch.float8_e4m3fn' in dtypes:
            info['weight_dtype'] = 'FP8_E4M3'
        elif 'torch.float8_e5m2' in dtypes:
            info['weight_dtype'] = 'FP8_E5M2'
        else:
            info['weight_dtype'] = list(dtypes)[0] if dtypes else 'unknown'
    else:
        # 检查普通权重中是否有量化类型
        for w in result['normal_weights']:
            if w['dtype'] in ['torch.int8', 'torch.int4', 'torch.uint8']:
                info['is_quantized'] = True
                info['weight_dtype'] = w['dtype'].replace('torch.', '').upper()
                break

    if not info['is_quantized']:
        info['summary'] = '非量化模型 (FP32/FP16/BF16)'
        return info

    # 判断量化粒度
    if result['weight_scales']:
        sample_scale = result['weight_scales'][0]
        shape = sample_scale['shape']

        if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
            info['quant_granularity'] = 'Per-Tensor'
        elif len(shape) == 2 and shape[1] == 1:
            info['quant_granularity'] = 'Per-Channel (Per-Row)'
        elif len(shape) == 2 and shape[1] > 1:
            info['quant_granularity'] = f'Per-Group (group_size≈{shape[1]})'
        else:
            info['quant_granularity'] = f'Unknown (shape={shape})'

    # 判断静态/动态量化
    if result['input_scales']:
        info['quant_type'] = 'Static (有预计算的 input_scale)'
    else:
        info['quant_type'] = 'Dynamic (无 input_scale，运行时计算)'

    # 判断对称/非对称量化
    if result['zero_points']:
        info['symmetry'] = 'Asymmetric (有 zero_point)'
    else:
        info['symmetry'] = 'Symmetric (无 zero_point)'

    # 生成总结
    info['summary'] = f"{info['weight_dtype']} {info['quant_granularity']} {info['quant_type']} {info['symmetry']}"

    return info


def print_report(model_path: str, result: Dict[str, Any], quant_info: Dict[str, str]):
    """打印分析报告"""

    print("\n" + "=" * 70)
    print(f"模型量化分析报告")
    print("=" * 70)
    print(f"\n模型路径: {model_path}")
    print(f"总张量数: {result['total_tensors']}")

    print("\n--- 数据类型统计 ---")
    for dtype, count in sorted(result['dtype_counts'].items()):
        print(f"  {dtype}: {count} 个张量")

    print("\n--- 量化类型判断 ---")
    print(f"  是否量化: {'是' if quant_info['is_quantized'] else '否'}")
    if quant_info['is_quantized']:
        print(f"  权重类型: {quant_info['weight_dtype']}")
        print(f"  量化粒度: {quant_info['quant_granularity']}")
        print(f"  量化类型: {quant_info['quant_type']}")
        print(f"  对称性:   {quant_info['symmetry']}")

    print(f"\n>>> 总结: {quant_info['summary']}")

    # 详细信息
    if result['quant_weights']:
        print(f"\n--- 量化权重样例 (共 {len(result['quant_weights'])} 个) ---")
        for w in result['quant_weights'][:5]:
            print(f"  {w['name']}: {w['dtype']}, shape={w['shape']}")
        if len(result['quant_weights']) > 5:
            print(f"  ... 还有 {len(result['quant_weights']) - 5} 个")

    if result['weight_scales']:
        print(f"\n--- 权重 Scales 样例 (共 {len(result['weight_scales'])} 个) ---")
        for s in result['weight_scales'][:5]:
            print(f"  {s['name']}: {s['dtype']}, shape={s['shape']}")
        if len(result['weight_scales']) > 5:
            print(f"  ... 还有 {len(result['weight_scales']) - 5} 个")

    if result['input_scales']:
        print(f"\n--- 输入 Scales (静态量化标志, 共 {len(result['input_scales'])} 个) ---")
        for s in result['input_scales'][:5]:
            print(f"  {s['name']}: {s['dtype']}, shape={s['shape']}")
        if len(result['input_scales']) > 5:
            print(f"  ... 还有 {len(result['input_scales']) - 5} 个")

    if result['zero_points']:
        print(f"\n--- Zero Points (非对称量化标志, 共 {len(result['zero_points'])} 个) ---")
        for z in result['zero_points'][:5]:
            print(f"  {z['name']}: {z['dtype']}, shape={z['shape']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='分析模型量化类型')
    parser.add_argument('model_path', type=str,
                        help='模型文件路径 (pytorch_model.bin 或 model.safetensors) 或模型目录')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示所有张量信息')

    args = parser.parse_args()

    # 处理路径
    model_path = args.model_path
    if os.path.isdir(model_path):
        # 如果是目录，尝试找模型文件
        candidates = [
            'pytorch_model.bin',
            'model.safetensors',
            'pytorch_model-00001-of-00001.bin',
        ]
        for c in candidates:
            full_path = os.path.join(model_path, c)
            if os.path.exists(full_path):
                model_path = full_path
                break
        else:
            # 尝试找任意 .bin 或 .safetensors 文件
            for f in os.listdir(model_path):
                if f.endswith('.bin') or f.endswith('.safetensors'):
                    model_path = os.path.join(model_path, f)
                    break

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return 1

    print(f"正在分析: {model_path}")

    result = analyze_model(model_path)
    quant_info = determine_quant_type(result)
    print_report(model_path, result, quant_info)

    if args.verbose:
        print("\n--- 所有张量 ---")
        state_dict = torch.load(model_path, map_location='cpu')
        for name, tensor in state_dict.items():
            print(f"  {name}: {tensor.dtype}, shape={list(tensor.shape)}")

    return 0


if __name__ == '__main__':
    exit(main())

# # 分析单个模型文件
# python3 analyze_quant_model.py /path/to/pytorch_model.bin

# # 分析模型目录（自动找模型文件）
# python3 analyze_quant_model.py /mnt/nas1/dm/bert_static_quant_2/quant/

# # 显示所有张量详情
# python3 analyze_quant_model.py /path/to/model --verbose