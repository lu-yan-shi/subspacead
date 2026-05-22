"""
SubspaceAnomalyDetector 快速使用示例
"""

from subspace_anomaly_detector import SubspaceAnomalyDetector, create_detector


def example_1_basic():
    """基础用法"""
    print("=" * 60)
    print("示例 1: 基础用法")
    print("=" * 60)
    
    # 创建检测器（自动使用本地模型）
    detector = SubspaceAnomalyDetector(
        image_res=384,
    )
    
    # 训练
    detector.train(["datas/template.jpg"])
    
    # 检测
    results = detector.detect(["datas/test-1.jpg", "datas/test-2.jpg"])
    
    for result in results:
        print(f"{result['image_name']}: {result['anomaly_score']:.4f}")


def example_2_quick():
    """快速创建和检测"""
    print("\n" + "=" * 60)
    print("示例 2: 快速检测")
    print("=" * 60)
    
    detector = create_detector(model=None, resolution=384)  # None 表示使用本地模型
    detector.train(["datas/template.jpg"])
    
    score = detector.detect_single("datas/test-1.jpg")
    print(f"异常分数：{score:.4f}")


def example_3_heatmap():
    """返回热力图"""
    print("\n" + "=" * 60)
    print("示例 3: 获取热力图")
    print("=" * 60)
    
    detector = SubspaceAnomalyDetector(image_res=384)
    detector.train(["datas/template.jpg"])
    
    score, heatmap = detector.detect_single(
        "datas/test-1.jpg",
        return_heatmap=True,
    )
    
    print(f"分数：{score:.4f}")
    print(f"热力图形状：{heatmap.shape}")


def example_4_custom_threshold():
    """自定义阈值"""
    print("\n" + "=" * 60)
    print("示例 4: 自定义阈值")
    print("=" * 60)
    
    detector = SubspaceAnomalyDetector(image_res=384)
    detector.train(["datas/template.jpg"])
    detector.set_threshold(0.25)
    
    score = detector.detect_single("datas/test-1.jpg")
    is_anomaly = detector.is_anomaly(score)
    
    print(f"分数：{score:.4f}, 是否异常：{is_anomaly}")


def example_5_export_load():
    """模型导出和加载"""
    print("\n" + "=" * 60)
    print("示例 5: 模型导出/加载")
    print("=" * 60)
    
    # 训练并导出
    detector1 = SubspaceAnomalyDetector(image_res=384)
    detector1.train(["datas/template.jpg"])
    detector1.export_model("test_model.pth")
    
    # 加载并使用
    detector2 = SubspaceAnomalyDetector()
    detector2.load_model("test_model.pth")
    
    results = detector2.detect(["datas/test-1.jpg"])
    print(f"加载模型后的检测结果：{results[0]['anomaly_score']:.4f}")
    
    # 清理
    import os
    if os.path.exists("test_model.pth"):
        os.remove("test_model.pth")


if __name__ == "__main__":
    try:
        example_1_basic()
        example_2_quick()
        example_3_heatmap()
        example_4_custom_threshold()
        example_5_export_load()
        
        print("\n✨ 所有示例运行完成!")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到：{e}")
        print("请确保 datas/ 目录下有测试图像")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
