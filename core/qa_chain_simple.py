import openai
from core.vector_store_compatible import vector_store
from config.settings import settings
from typing import Dict, List
import time
import json

class SimpleRAGChain:
    """简化的RAG问答链条"""
    
    def __init__(self):
        """初始化RAG链条"""
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # 提示模板
        self.system_prompt = """你是一个专业的AI助手。请基于以下文档内容回答用户的问题。

请遵循以下要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁明了，突出重点
4. 可以适当引用文档中的原文

相关文档内容：
{context}

用户问题：{question}

请提供准确、有用的回答："""
    
    def calculate_confidence(self, source_docs, question, answer):
        """改进的置信度计算"""
        if not source_docs:
            return 0.1
        
        # 1. 基础分数：基于检索到的文档数量
        doc_score = min(len(source_docs) / settings.TOP_K, 1.0) * 0.4
        
        # 2. 相关性分数：基于文档内容与问题的匹配度
        relevance_score = 0
        question_words = set(question.lower().split())
        
        for doc in source_docs:
            doc_words = set(doc.page_content.lower().split())
            # 计算词汇重叠度
            if question_words and doc_words:
                overlap = len(question_words & doc_words) / len(question_words | doc_words)
                relevance_score += overlap
        
        relevance_score = min(relevance_score / len(source_docs), 1.0) * 0.3 if source_docs else 0
        
        # 3. 答案长度分数：太短的答案可能不够详细
        answer_length_score = min(len(answer.split()) / 50, 1.0) * 0.2
        
        # 4. 关键词匹配分数：答案是否包含问题中的关键词
        keyword_score = 0
        for word in question_words:
            if len(word) > 2 and word in answer.lower():
                keyword_score += 1
        keyword_score = min(keyword_score / max(len(question_words), 1), 1.0) * 0.1
        
        total_confidence = doc_score + relevance_score + answer_length_score + keyword_score
        
        return min(max(total_confidence, 0.1), 0.95)  # 限制在0.1-0.95之间
    
    def ask(self, question: str) -> Dict:
        """提问并获取答案"""
        print(f"🤔 用户问题: {question}")
        
        start_time = time.time()
        
        try:
            # 1. 检索相关文档
            print("正在检索相关文档...")
            source_docs = vector_store.similarity_search(question, k=settings.TOP_K)
            
            # 2. 构建上下文
            context_parts = []
            citations = []
            
            for i, doc in enumerate(source_docs):
                context_parts.append(f"文档片段 {i+1}:\n{doc.page_content}")
                
                citations.append({
                    "chunk_id": i,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_id", i)
                })
            
            context = "\n\n".join(context_parts)
            
            # 3. 构建完整提示
            full_prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            print(f"构建上下文完成，使用了 {len(source_docs)} 个文档片段")
            
            # 4. 调用LLM生成答案
            print(" 正在生成答案...")
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # 5. 计算响应时间
            response_time = time.time() - start_time
            
            # 6. 计算置信度
            confidence = self.calculate_confidence(source_docs, question, answer)
            
            # 7. 获取token使用量
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            result_dict = {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs),
                "tokens_used": tokens_used
            }
            
            print(f"回答生成完成")
            print(f"总耗时: {response_time:.2f} 秒")
            print(f"使用了 {len(source_docs)} 个文档片段")
            if tokens_used:
                print(f"消耗 tokens: {tokens_used}")
            
            return result_dict
            
        except Exception as e:
            print(f"问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0,
                "tokens_used": 0
            }
    
    def interactive_chat(self):
        """交互式聊天模式"""
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
                
                result = self.ask(question)
                print(f"\n 回答: {result['answer']}")
                print(f"置信度: {result['confidence']:.2f}")
                
                if result['citations']:
                    print("\n参考文档:")
                    for i, citation in enumerate(result['citations'], 1):
                        print(f"   {i}. {citation['content']}")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"出现错误: {e}")

# 创建全局RAG链条实例
rag_chain = SimpleRAGChain()
