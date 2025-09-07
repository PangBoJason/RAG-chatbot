from core.qa_chain_simple import rag_chain
from database.db_manager import db_manager
from config.settings import settings

def analyze_confidence(result):
    """分析置信度并给出建议"""
    confidence = result['confidence']
    
    if confidence >= 0.8:
        level = "高 高"
        advice = "答案质量很好，可以直接使用"
    elif confidence >= 0.6:
        level = "中 中"
        advice = "答案基本可靠，建议交叉验证"
    elif confidence >= 0.4:
        level = "🟠 低"
        advice = "答案参考价值有限，需要更多信息"
    else:
        level = "低 很低"
        advice = "建议重新组织问题或添加更多文档"
    
    return level, advice

def test_qa_chain():
    """测试RAG问答链条"""
    print("=== RAG 问答链条测试 ===\n")
    
    # 创建会话
    session_id = db_manager.create_conversation("测试用户")
    print(f"创建会话: {session_id}")
    
    # 测试问题列表（包括一些可能置信度较低的问题）
    test_questions = [
        "什么是RAG？",                    # 应该有高置信度
        "RAG有什么优势？",                # 应该有高置信度
        "RAG的工作流程是什么？",          # 应该有中等置信度
        "RAG可以用在哪些场景？",          # 应该有中等置信度
        "如何训练一个深度学习模型？",      # 应该有低置信度（文档中没有相关内容）
    ]
    
    confidence_stats = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} 问题 {i} {'='*20}")
        
        # 获取答案
        result = rag_chain.ask(question)
        
        # 分析置信度
        confidence_level, advice = analyze_confidence(result)
        confidence_stats.append(result['confidence'])
        
        # 显示结果
        print(f"\n❓ 问题: {result['question']}")
        print(f" 答案: {result['answer']}")
        print(f"置信度: {result['confidence']:.3f} ({confidence_level})")
        print(f"建议: {advice}")
        print(f"响应时间: {result['response_time']:.2f}秒")
        
        if result['tokens_used']:
            print(f"消耗tokens: {result['tokens_used']}")
        
        print(f"\n引用片段 ({len(result['citations'])}个):")
        for j, citation in enumerate(result['citations'], 1):
            print(f"   {j}. 来源: {citation['source']}")
            print(f"      内容: {citation['content'][:100]}...")
        
        # 记录到数据库
        qa_id = db_manager.log_qa(
            session_id=session_id,
            question=result['question'],
            answer=result['answer'],
            citations=result['citations'],
            confidence=result['confidence'],
            response_time=result['response_time'],
            tokens_used=result['tokens_used'],
            model_name=settings.MODEL_NAME
        )
        
        print(f"问答记录ID: {qa_id}")
    
    # 置信度统计分析
    print(f"\n{'='*50}")
    print("置信度统计分析:")
    print(f"   平均置信度: {sum(confidence_stats)/len(confidence_stats):.3f}")
    print(f"   最高置信度: {max(confidence_stats):.3f}")
    print(f"   最低置信度: {min(confidence_stats):.3f}")
    
    high_conf_count = sum(1 for c in confidence_stats if c >= 0.8)
    medium_conf_count = sum(1 for c in confidence_stats if 0.6 <= c < 0.8)
    low_conf_count = sum(1 for c in confidence_stats if c < 0.6)
    
    print(f"   高置信度(≥0.8): {high_conf_count} 个")
    print(f"   中置信度(0.6-0.8): {medium_conf_count} 个")
    print(f"   低置信度(<0.6): {low_conf_count} 个")
    
    # 显示会话历史
    print(f"\n会话历史:")
    history = db_manager.get_conversation_history(session_id)
    
    for i, qa in enumerate(history, 1):
        confidence_level, _ = analyze_confidence(qa)
        print(f"  问题{i}: {qa['question'][:30]}...")
        print(f"  置信度: {qa['confidence']:.3f} ({confidence_level})")
        print()
    
    print(f"RAG问答链条测试完成！")
    print(f"总计 {len(test_questions)} 个问答已记录到数据库")

def test_interactive():
    """测试交互式聊天（带置信度提示）"""
    print("=== 交互式聊天测试 (带置信度分析) ===")
    print(" RAG助手已启动！输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n👤 您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not question:
                continue
            
            result = rag_chain.ask(question)
            confidence_level, advice = analyze_confidence(result)
            
            print(f"\n 回答: {result['answer']}")
            print(f"置信度: {result['confidence']:.3f} ({confidence_level})")
            print(f"建议: {advice}")
            
            if result['citations']:
                print("\n参考文档:")
                for i, citation in enumerate(result['citations'], 1):
                    print(f"   {i}. {citation['content'][:100]}...")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"出现错误: {e}")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 批量测试（包含置信度分析）")
    print("2. 交互式聊天（带置信度提示）")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "2":
        test_interactive()
    else:
        test_qa_chain()
