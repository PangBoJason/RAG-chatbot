"""
HyDE (Hypothetical Document Embeddings) 检索增强模块
通过生成假设性答案来改善检索效果
"""
import openai
from core.vector_store_compatible import vector_store
from config.settings import settings
from typing import List, Dict
from langchain.schema import Document

class HyDERetriever:
    """HyDE检索器"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # HyDE提示模板
        self.hyde_prompt = """请基于以下问题生成一个假设性的详细答案。这个答案将用于文档检索，所以请包含可能相关的关键词和概念。

问题: {question}

假设性答案:"""
    
    def generate_hypothetical_answer(self, question: str) -> str:
        """生成假设性答案"""
        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "user", "content": self.hyde_prompt.format(question=question)}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"生成假设性答案失败: {e}")
            return question  # fallback到原问题
    
    def hyde_retrieve(self, question: str, k: int = None) -> List[Document]:
        """使用HyDE方法检索文档"""
        k = k or settings.TOP_K
        
        print(f"🔍 使用HyDE增强检索...")
        
        # 1. 生成假设性答案（后台处理，不显示给用户）
        hypothetical_answer = self.generate_hypothetical_answer(question)
        
        # 2. 使用假设性答案进行检索
        hyde_results = vector_store.similarity_search(hypothetical_answer, k=k*2)
        
        # 3. 同时使用原问题检索
        question_results = vector_store.similarity_search(question, k=k)
        
        # 4. 合并和去重结果
        all_results = hyde_results + question_results
        unique_results = []
        seen_content = set()
        
        for doc in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
                if len(unique_results) >= k:
                    break
        
        print(f"HyDE检索完成，返回 {len(unique_results)} 个文档")
        return unique_results[:k]
    
    def compare_retrieval_methods(self, question: str, k: int = 5) -> Dict:
        """比较不同检索方法的效果"""
        print(f"比较检索方法，问题: {question}")
        
        # 基础检索
        basic_results = vector_store.similarity_search(question, k=k)
        
        # HyDE检索
        hyde_results = self.hyde_retrieve(question, k=k)
        
        # 生成假设性答案
        hypothetical_answer = self.generate_hypothetical_answer(question)
        
        return {
            "question": question,
            "hypothetical_answer": hypothetical_answer,
            "basic_results": basic_results,
            "hyde_results": hyde_results,
            "basic_count": len(basic_results),
            "hyde_count": len(hyde_results)
        }

# 创建全局HyDE检索器实例
hyde_retriever = HyDERetriever()
