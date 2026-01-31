"""
使用torchview生成Qwen3完整网络架构图

尝试多种配置以找到最适合大型Transformer模型的可视化方案
"""

import torch
import json
from pathlib import Path


def try_torchview_config(model, config_name, **kwargs):
    """
    尝试特定的torchview配置
    """
    try:
        from torchview import draw_graph

        print(f"\n{'='*80}")
        print(f"尝试配置: {config_name}")
        print(f"参数: {kwargs}")
        print('='*80)

        dummy_input = torch.randint(0, 1000, (1, 32), dtype=torch.long)

        model_graph = draw_graph(
            model,
            input_data=dummy_input,
            **kwargs
        )

        print(f"✓ 成功生成: {kwargs.get('filename', 'graph')}.gv.pdf")
        return True, model_graph

    except Exception as e:
        print(f"✗ 失败: {e}")
        return False, None


def generate_architecture_variations(model):
    """
    尝试多种配置生成架构图
    """
    print("="*80)
    print("Qwen3 完整网络架构可视化")
    print("="*80)

    from torchview import draw_graph

    # 只保留完整版配置
    configs = [
        {
            "name": "完整版（最详细）",
            "params": {
                "filename": "qwen3_full",
                "depth": 6,
                "expand_nested": True,
                "hide_inner_tensors": True,  # 隐藏张量细节避免过于复杂
                "hide_module_functions": False,
                "graph_name": "Qwen3 Full Architecture",
                "save_graph": True,
                "roll": True,
            }
        },
    ]

    successful = []
    failed = []

    for config in configs:
        success, graph = try_torchview_config(
            model,
            config["name"],
            **config["params"]
        )

        if success:
            successful.append(config["name"])
        else:
            failed.append(config["name"])

    print("\n" + "="*80)
    print("生成结果总结")
    print("="*80)
    print(f"\n✓ 成功 ({len(successful)}):")
    for name in successful:
        print(f"  - {name}")

    if failed:
        print(f"\n✗ 失败 ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")

    return successful


def main():
    """
    主函数 - 只生成qwen3_full完整架构图
    """
    print("Qwen3 完整架构图生成工具")
    print("="*80)

    # 加载模型
    config_path = Path(__file__).parent / "hg_config.json"

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        with open(config_path) as f:
            config_dict = json.load(f)

        config = AutoConfig.for_model(**config_dict)
        config._attn_implementation = "eager"

        print("正在加载模型...")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        # 生成完整架构图
        print("\n开始生成qwen3_full完整架构图...")
        successful = generate_architecture_variations(model)

        print("\n" + "="*80)
        print("完成！")
        print("="*80)

        if successful:
            print("\n✓ 生成的文件：")
            print("  - qwen3_full.gv.pdf - Qwen3完整架构图")
            print("\n使用PDF阅读器打开 qwen3_full.gv.pdf 查看")
        else:
            print("\n✗ 生成失败，请检查上述错误信息")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
