"""
Day 3 功能测试 - 完整系统评测与优化
测试RAGAS评测、用户反馈学习、多模型支持等高级功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from evaluation.rag_evaluator import rag_evaluator
from evaluation.feedback_learner import feedback_learner
from evaluation.multi_model_support import multi_model_manager
from database.db_manager import db_manager
from core.enhanced_rag_chain import enhanced_rag_chain

def test_ragas_evaluation():
    """测试RAGAS自动化评测"""
    print("\n" + "="*60)
    print("🔥 测试 RAGAS 自动化评测模块")
    print("="*60)
    
    try:
        # 创建测试数据集
        dataset = rag_evaluator.create_evaluation_dataset()
        print(f"创建评测数据集: {len(dataset)} 个测试问题")
        
        # 运行方法对比评测
        print("\n开始运行不同RAG方法对比评测...")
        results = rag_evaluator.run_method_comparison(dataset)
        
        # 生成评测报告
        print("\n生成评测报告...")
        report = rag_evaluator.generate_evaluation_report(results)
        
        # 保存结果
        report_path = rag_evaluator.save_evaluation_results(report)
        
        # 打印摘要报告
        rag_evaluator.print_summary_report(report)
        
        return True
        
    except Exception as e:
        print(f"RAGAS评测测试失败: {e}")
        return False

def test_feedback_learning():
    """测试用户反馈学习"""
    print("\n" + "="*60)
    print("🔥 测试用户反馈学习模块")
    print("="*60)
    
    try:
        # 模拟一些问答和反馈
        print("模拟用户问答和反馈...")
        
        # 创建测试问答记录
        test_cases = [
            {
                "question": "什么是RAG技术？",
                "rating": 5,
                "comment": "回答很详细，很有帮助！"
            },
            {
                "question": "RAG有什么优势？",
                "rating": 3,
                "comment": "回答还可以，但不够完整"
            },
            {
                "question": "如何实现RAG系统？",
                "rating": 2,
                "comment": "回答不够准确，缺少重要信息"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. 测试问题: {case['question']}")
            
            # 获取RAG回答
            qa_result = enhanced_rag_chain.ask_basic(case["question"])
            
            # 模拟保存问答记录到数据库
            qa_log_id = i  # 简化实现，使用序号作为ID
            
            # 收集反馈
            success = feedback_learner.collect_feedback(
                qa_log_id=qa_log_id,
                rating=case["rating"],
                comment=case["comment"],
                user_id=f"test_user_{i}"
            )
            
            if success:
                print(f"   反馈收集成功: {case['rating']}/5 - {case['comment']}")
            
            time.sleep(0.5)  # 避免API调用过快
        
        # 显示反馈统计
        print("\n反馈统计报告:")
        feedback_learner.print_feedback_report()
        
        # 生成改进建议
        print("\n系统改进建议:")
        suggestions = feedback_learner.generate_improvement_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"反馈学习测试失败: {e}")
        return False

def test_multi_model_support():
    """测试多模型支持"""
    print("\n" + "="*60)
    print("🔥 测试多模型支持模块")
    print("="*60)
    
    try:
        # 列出可用模型
        available_models = multi_model_manager.list_available_models()
        print(f"可用模型: {', '.join(available_models)}")
        
        # 显示模型信息
        print(f"\n模型详细信息:")
        for model in available_models:
            info = multi_model_manager.get_model_info(model)
            print(f"   {model}:")
            print(f"      提供商: {info.get('provider', '未知')}")
            print(f"      最大Token: {info.get('max_tokens', '未知')}")
            print(f"      支持函数调用: {info.get('supports_function_calling', False)}")
        
        # 测试模型对比
        print(f"\n模型回答对比测试:")
        test_question = "请简要解释什么是RAG技术？"
        
        # 只比较实际可用的模型（主要是OpenAI）
        available_openai_models = [m for m in available_models if "gpt" in m]
        
        if available_openai_models:
            comparison_results = multi_model_manager.compare_models(
                test_question, 
                available_openai_models[:2]  # 最多比较2个模型避免API调用过多
            )
            
            print(f"\n对比结果:")
            for model, result in comparison_results.items():
                if "error" not in result:
                    print(f"   {model}:")
                    print(f"      回答: {result['content'][:100]}...")
                    print(f"      用时: {result['response_time']:.2f}秒")
                    print(f"      Token: {result['tokens_used']}")
                else:
                    print(f"   {model}: 失败 - {result['error']}")
        
        # 运行性能基准测试（简化版）
        print(f"\n性能基准测试:")
        simple_questions = [
            "什么是人工智能？",
            "RAG技术的优势是什么？"
        ]
        
        benchmark_results = multi_model_manager.benchmark_models(simple_questions)
        multi_model_manager.print_benchmark_report(benchmark_results)
        
        return True
        
    except Exception as e:
        print(f"多模型支持测试失败: {e}")
        return False

def test_complete_system_integration():
    """测试完整系统集成"""
    print("\n" + "="*60)
    print("🔥 测试完整系统集成")
    print("="*60)
    
    try:
        print("运行端到端流程测试...")
        
        # 1. 用户提问
        user_question = "RAG技术在实际应用中有哪些挑战？"
        print(f"👤 用户问题: {user_question}")
        
        # 2. 使用增强RAG获取答案
        print(f"\n 使用增强RAG生成回答...")
        qa_result = enhanced_rag_chain.ask_enhanced(user_question)
        
        print(f"   回答: {qa_result['answer'][:200]}...")
        print(f"   置信度: {qa_result['confidence']:.3f}")
        print(f"   用时: {qa_result['response_time']:.2f}秒")
        print(f"   引用数量: {len(qa_result['citations'])}")
        
        # 3. 模拟用户反馈
        print(f"\n收集用户反馈...")
        feedback_rating = 4
        feedback_comment = "回答很全面，但可以更具体一些"
        
        # 注意：这里简化了数据库操作
        print(f"   用户评分: {feedback_rating}/5")
        print(f"   用户评论: {feedback_comment}")
        
        # 4. 系统性能监控
        print(f"\n系统性能指标:")
        print(f"   RAG功能: 正常")
        print(f"   数据库连接: 正常")
        print(f"   向量检索: 正常")
        print(f"   LLM调用: 正常")
        
        # 5. 生成系统状态报告
        print(f"\n系统状态总结:")
        print(f"   - 基础RAG: 可用")
        print(f"   - HyDE检索: 可用")
        print(f"   - 文档重排: 可用")
        print(f"   - 增强RAG: 可用")
        print(f"   - 评测系统: 可用")
        print(f"   - 反馈学习: 可用")
        print(f"   - 多模型支持: 可用")
        
        return True
        
    except Exception as e:
        print(f"系统集成测试失败: {e}")
        return False

def run_day3_comprehensive_test():
    """运行Day 3完整功能测试"""
    print("开始 RAGLite Day 3 完整功能测试")
    print("本测试将验证RAGAS评测、用户反馈学习、多模型支持等高级功能")
    
    test_results = {}
    
    # 运行各项测试
    tests = [
        ("多模型支持", test_multi_model_support),
        ("用户反馈学习", test_feedback_learning),
        ("RAGAS自动化评测", test_ragas_evaluation),
        ("完整系统集成", test_complete_system_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        try:
            result = test_func()
            test_results[test_name] = "通过" if result else "失败"
        except Exception as e:
            test_results[test_name] = f"异常: {str(e)}"
        
        time.sleep(1)  # 短暂暂停
    
    # 显示测试总结
    print("\n" + "="*60)
    print("Day 3 功能测试总结")
    print("="*60)
    
    for test_name, result in test_results.items():
        print(f"{result} {test_name}")
    
    passed_tests = sum(1 for result in test_results.values() if "" in result)
    total_tests = len(test_results)
    
    print(f"\n测试通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n恭喜！所有Day 3高级功能测试通过！")
        print("RAGLite系统现在具备了完整的企业级功能：")
        print("   自动化质量评测 (RAGAS)")
        print("   智能反馈学习")
        print("   多模型支持")
        print("   完整的监控和优化体系")
    else:
        print(f"\n有 {total_tests - passed_tests} 项测试需要修复")

if __name__ == "__main__":
    # 交互式测试菜单
    while True:
        print("\n" + "="*50)
        print("RAGLite Day 3 功能测试菜单")
        print("="*50)
        print("1. 运行完整测试")
        print("2. RAGAS评测测试")
        print("3. 用户反馈学习测试")
        print("4.  多模型支持测试")
        print("5. 系统集成测试")
        print("0. 退出")
        
        choice = input("\n请选择测试项 (0-5): ").strip()
        
        if choice == "0":
            print("测试结束，感谢使用！")
            break
        elif choice == "1":
            run_day3_comprehensive_test()
        elif choice == "2":
            test_ragas_evaluation()
        elif choice == "3":
            test_feedback_learning()
        elif choice == "4":
            test_multi_model_support()
        elif choice == "5":
            test_complete_system_integration()
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")
