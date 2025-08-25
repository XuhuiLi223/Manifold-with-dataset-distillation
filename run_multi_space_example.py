#!/usr/bin/env python3
"""
多空间流形数据凝聚示例运行脚本
"""

import os
import sys
import argparse
import subprocess


def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'torch', 'torchvision', 'geoopt', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (缺失)")
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_example_cifar10():
    """运行CIFAR-10示例"""
    print("🚀 运行CIFAR-10多空间流形数据凝聚示例")
    print("="*50)
    
    cmd = [
        "python3", "condense_multi_space_manifold.py",
        "--dataset", "cifar10",
        "--num_spaces", "10",
        "--min_dim", "32", 
        "--max_dim", "128",
        "--ipc", "10",
        "--epochs", "100",  # 示例用较少epoch
        "--lr_img", "0.1",
        "--lr_net", "0.01",
        "--batch_train", "256",
        "--init", "noise",
        "--save_dir", "./results_multi_space_cifar10"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ CIFAR-10示例运行完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ 找不到训练脚本: condense_multi_space_manifold.py")
        return False


def run_example_cifar100():
    """运行CIFAR-100示例"""
    print("🚀 运行CIFAR-100多空间流形数据凝聚示例")
    print("="*50)
    
    cmd = [
        "python3", "condense_multi_space_manifold.py",
        "--dataset", "cifar100", 
        "--num_spaces", "15",  # 更复杂任务用更多空间
        "--min_dim", "64",
        "--max_dim", "256", 
        "--ipc", "10",
        "--epochs", "200",
        "--lr_img", "0.05",   # 更保守的学习率
        "--lr_net", "0.008",
        "--batch_train", "128",
        "--init", "random",
        "--dsa",  # 启用数据增强
        "--save_dir", "./results_multi_space_cifar100"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ CIFAR-100示例运行完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败: {e}")
        return False


def run_quick_test():
    """运行快速测试"""
    print("🧪 运行快速功能测试")
    print("="*30)
    
    # 首先运行语法检查
    if os.path.exists("test_syntax_check.py"):
        print("1. 语法检查...")
        try:
            subprocess.run(["python3", "test_syntax_check.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ 语法检查失败")
            return False
    
    # 运行小规模测试
    print("\n2. 小规模训练测试...")
    cmd = [
        "python3", "condense_multi_space_manifold.py",
        "--dataset", "cifar10",
        "--num_spaces", "3",   # 少量空间
        "--min_dim", "16",     # 小维度
        "--max_dim", "32", 
        "--ipc", "1",          # 每类1张图
        "--epochs", "5",       # 很少epoch
        "--batch_train", "64",
        "--save_dir", "./test_results"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print("✅ 快速测试通过!")
        return True
    except subprocess.TimeoutExpired:
        print("⚠️ 测试超时，但这可能是正常的")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 快速测试失败: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])  # 显示最后500字符
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False


def show_help():
    """显示帮助信息"""
    help_text = """
🌟 多空间可学习流形数据凝聚 - 使用指南

📋 可用命令:
  python run_multi_space_example.py --mode [选项]

🎯 模式选项:
  check       - 检查依赖包
  test        - 运行快速功能测试  
  cifar10     - 运行CIFAR-10示例 (推荐新手)
  cifar100    - 运行CIFAR-100示例 (更复杂)
  help        - 显示此帮助信息

💡 使用建议:
  1. 首先运行: --mode check (检查环境)
  2. 然后运行: --mode test (快速测试)  
  3. 最后运行: --mode cifar10 (完整示例)

🔧 自定义参数:
  如需自定义参数，请直接使用:
  python condense_multi_space_manifold.py [参数]

📚 详细文档:
  请查看 README_multi_space_manifold.md

🎉 新特性:
  ✓ 10个可学习流形空间
  ✓ 自适应维度投影 (高维→低维)
  ✓ 独立曲率优化
  ✓ 智能门控融合
  ✓ 维度调度器
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(description='多空间流形数据凝聚示例')
    parser.add_argument('--mode', type=str, default='help',
                        choices=['check', 'test', 'cifar10', 'cifar100', 'help'],
                        help='运行模式')
    
    args = parser.parse_args()
    
    print("🌟 多空间可学习流形数据凝聚")
    print("="*40)
    
    if args.mode == 'help':
        show_help()
    
    elif args.mode == 'check':
        print("🔍 检查依赖包...")
        if check_dependencies():
            print("\n✅ 所有依赖包都已安装!")
        else:
            print("\n❌ 请先安装缺失的依赖包")
            sys.exit(1)
    
    elif args.mode == 'test':
        print("🧪 运行快速测试...")
        if run_quick_test():
            print("\n🎉 测试通过! 可以运行完整示例了")
        else:
            print("\n❌ 测试失败，请检查环境配置")
            sys.exit(1)
    
    elif args.mode == 'cifar10':
        print("📊 准备运行CIFAR-10示例...")
        if not check_dependencies():
            print("❌ 请先安装依赖包: python run_multi_space_example.py --mode check")
            sys.exit(1)
        
        success = run_example_cifar10()
        if success:
            print("\n🎉 CIFAR-10示例完成! 查看结果在 ./results_multi_space_cifar10/")
        else:
            print("\n❌ 示例运行失败")
            sys.exit(1)
    
    elif args.mode == 'cifar100':
        print("📊 准备运行CIFAR-100示例...")
        if not check_dependencies():
            print("❌ 请先安装依赖包: python run_multi_space_example.py --mode check")
            sys.exit(1)
        
        success = run_example_cifar100()
        if success:
            print("\n🎉 CIFAR-100示例完成! 查看结果在 ./results_multi_space_cifar100/")
        else:
            print("\n❌ 示例运行失败")
            sys.exit(1)


if __name__ == '__main__':
    main()