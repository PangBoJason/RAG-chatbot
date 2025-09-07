"""
Day 2 增强功能测试脚本
测试HyDE检索和Rerank重排序
"""
from core.enhanced_rag_chain import enhanced_rag_chain
from core.hyde_retrieval import hyde_retriever
from core.reranker import reranker
from database.db_manager import db_manager
from config.settings import settings
import time

def test_hyde_retrieval():
    """测试HyDE检索功能"""
    print("=" * 60)
    print("测试HyDE检索功能")
    print("=" * 60)
    
    test_questions = [
        "RAG技术的具体实现步骤",
        "如何提升检索增强生成的准确性",
        "向量数据库在AI中的作用"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} 问题 {i} {'='*20}")
        print(f"问题: {question}")
        
        # 比较基础检索和HyDE检索
        comparison = hyde_retriever.compare_retrieval_methods(question, k=3)
        
        print(f"\n假设性答案:")
        print(f"   {comparison['hypothetical_answer'][:150]}...")
        
        print(f"\n检索结果对比:")
        print(f"   基础检索: {comparison['basic_count']} 个结果")
        print(f"   HyDE检索: {comparison['hyde_count']} 个结果")
        
        print(f"\n🔵 基础检索结果:")
        for j, doc in enumerate(comparison['basic_results'][:2], 1):
            print(f"   {j}. {doc.page_content[:100]}...")
        
        print(f"\n中 HyDE检索结果:")
        for j, doc in enumerate(comparison['hyde_results'][:2], 1):
            print(f"   {j}. {doc.page_content[:100]}...")

def test_rerank_functionality():
    """测试重排序功能"""
    print("=" * 60)
    print("测试重排序功能")
    print("=" * 60)
    
    test_questions = [
        "RAG有哪些优势",
        "如何实现文档检索",
        "向量化的作用是什么"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} 问题 {i} {'='*20}")
        print(f"问题: {question}")
        
        # 获取基础检索结果
        from core.vector_store_compatible import vector_store
        documents = vector_store.similarity_search(question, k=6)
        
        if documents:
            # 比较排序方法
            comparison = reranker.compare_ranking_methods(question, documents, top_k=3)
            
            print(f"\n排序方法对比:")
            print(f"   原始文档数: {comparison['documents_count']}")
            
            print(f"\n🔵 原始顺序 (向量相似度):")
            for j, (doc, score) in enumerate(comparison['original_order'][:3], 1):
                print(f"   {j}. {doc.page_content[:80]}...")
            
            print(f"\n🟠 重排序后:")
            for j, (doc, score) in enumerate(comparison['simple_ranking'][:3], 1):
                print(f"   {j}. 分数:{score:.3f} | {doc.page_content[:80]}...")
        else:
            print("没有找到相关文档")

def test_enhanced_rag_methods():
    """测试增强版RAG方法"""
    print("=" * 60)
    print("测试增强版RAG方法")
    print("=" * 60)
    
    test_questions = [
        "什么是RAG技术？",
        "RAG有什么优势？",
        "如何实现高质量的文档问答？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*30} 问题 {i} {'='*30}")
        print(f"问题: {question}")
        
        # 比较所有方法
        comparison = enhanced_rag_chain.compare_methods(question)
        
        print(f"\n方法效果对比:")
        print(f"置信度排名:")
        for rank, (method, confidence) in enumerate(comparison['comparison']['confidence_ranking'], 1):
            print(f"   {rank}. {method}: {confidence:.3f}")
        
        print(f"\n速度排名:")
        for rank, (method, time_cost) in enumerate(comparison['comparison']['speed_ranking'], 1):
            print(f"   {rank}. {method}: {time_cost:.2f}秒")
        
        print(f"\n各方法回答对比:")
        for method, result in comparison['results'].items():
            print(f"\n{method.upper()} 方法:")
            print(f"   回答: {result['answer'][:120]}...")
            print(f"   置信度: {result['confidence']:.3f}")
            print(f"   用时: {result['response_time']:.2f}秒")
            if result['tokens_used']:
                print(f"   消耗tokens: {result['tokens_used']}")

def test_database_integration():
    """测试数据库集成"""
    print("=" * 60)
    print("💾 测试增强功能的数据库集成")
    print("=" * 60)
    
    # 创建测试会话
    session_id = db_manager.create_conversation("enhanced_test_user")
    print(f"创建测试会话: {session_id}")
    
    test_questions = [
        "使用基础方法：RAG是什么？",
        "使用HyDE方法：RAG的应用场景？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试问题 {i} ---")
        
        if "基础方法" in question:
            clean_question = question.replace("使用基础方法：", "")
            result = enhanced_rag_chain.ask_basic(clean_question)
        else:
            clean_question = question.replace("使用HyDE方法：", "")
            result = enhanced_rag_chain.ask_hyde(clean_question)
        
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
        print(f"   方法: {result['retrieval_method']}")
        print(f"   置信度: {result['confidence']:.3f}")
        print(f"   用时: {result['response_time']:.2f}秒")
    
    # 显示会话历史
    print(f"\n会话历史:")
    history = db_manager.get_conversation_history(session_id)
    for i, qa in enumerate(history, 1):
        print(f"  问答{i}: {qa['question'][:30]}...")
        print(f"  置信度: {qa['confidence']:.3f}")

def interactive_enhanced_chat():
    """交互式增强聊天"""
    print("=" * 60)
    print(" 增强版RAG助手 - 交互模式")
    print("=" * 60)
    print("可用命令:")
    print("  basic: <问题>     - 使用基础检索")
    print("  hyde: <问题>      - 使用HyDE检索")
    print("  rerank: <问题>    - 使用重排序")
    print("  enhanced: <问题>  - 使用HyDE+重排序")
    print("  compare: <问题>   - 比较所有方法")
    print("  quit              - 退出")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 您的指令: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 解析命令
            if user_input.startswith('basic:'):
                question = user_input[6:].strip()
                result = enhanced_rag_chain.ask_basic(question)
                print(f"\n🔵 基础方法回答:")
                print(f" {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                
            elif user_input.startswith('hyde:'):
                question = user_input[5:].strip()
                result = enhanced_rag_chain.ask_hyde(question)
                print(f"\n中 HyDE方法回答:")
                print(f" {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                
            elif user_input.startswith('rerank:'):
                question = user_input[7:].strip()
                result = enhanced_rag_chain.ask_rerank(question)
                print(f"\n🟠 重排序方法回答:")
                print(f" {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                
            elif user_input.startswith('enhanced:'):
                question = user_input[9:].strip()
                result = enhanced_rag_chain.ask_enhanced(question)
                print(f"\n高 增强方法回答:")
                print(f" {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                
            elif user_input.startswith('compare:'):
                question = user_input[8:].strip()
                print(f"\n比较所有方法...")
                comparison = enhanced_rag_chain.compare_methods(question)
                
                print(f"\n置信度排名:")
                for rank, (method, confidence) in enumerate(comparison['comparison']['confidence_ranking'], 1):
                    print(f"   {rank}. {method}: {confidence:.3f}")
                
                print(f"\n最佳答案 ({comparison['comparison']['confidence_ranking'][0][0]}):")
                best_method = comparison['comparison']['confidence_ranking'][0][0]
                best_result = comparison['results'][best_method]
                print(f" {best_result['answer']}")
                
            else:
                print("无效命令，请使用正确的格式")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"出现错误: {e}")

def main():
    """主测试函数"""
    print("RAGLite Day 2 增强功能测试")
    print("选择测试模式:")
    print("1. 测试HyDE检索")
    print("2. 测试重排序功能")
    print("3. 测试增强版RAG方法")
    print("4. 测试数据库集成")
    print("5. 交互式增强聊天")
    print("6. 运行所有测试")
    
    choice = input("请选择 (1-6): ").strip()
    
    if choice == "1":
        test_hyde_retrieval()
    elif choice == "2":
        test_rerank_functionality()
    elif choice == "3":
        test_enhanced_rag_methods()
    elif choice == "4":
        test_database_integration()
    elif choice == "5":
        interactive_enhanced_chat()
    elif choice == "6":
        print("运行所有测试...")
        test_hyde_retrieval()
        time.sleep(2)
        test_rerank_functionality()
        time.sleep(2)
        test_enhanced_rag_methods()
        time.sleep(2)
        test_database_integration()
        print("\n所有测试完成！")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
