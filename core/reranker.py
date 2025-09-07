"""
Rerank 重排序模块
使用简化的相关性评分对检索结果进行重排序
"""
import openai
from config.settings import settings
from typing import List, Tuple, Dict
from langchain.schema import Document
import re

class SimpleReranker:
    """简化的重排序器"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
    
    def calculate_lexical_similarity(self, query: str, document: str) -> float:
        """计算词汇相似度"""
        # 简单的词汇重叠计算
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        if not query_words or not doc_words:
            return 0.0
        
        intersection = query_words & doc_words
        union = query_words | doc_words
        
        return len(intersection) / len(union)
    
    def calculate_semantic_score(self, query: str, document: str) -> float:
        """使用LLM计算语义相关性评分"""
        prompt = f"""
请评估以下文档内容与用户问题的相关性，给出0-10分的评分：

用户问题: {query}

文档内容: {document[:500]}...

评分标准:
- 10分: 完全相关，直接回答问题
- 8-9分: 高度相关，包含重要信息
- 6-7分: 中等相关，有一定参考价值
- 4-5分: 低度相关，包含少量相关信息
- 0-3分: 不相关或无关

请只返回数字评分(0-10):"""

        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            # 提取数字
            score_match = re.search(r'\d+', score_text)
            if score_match:
                score = float(score_match.group()) / 10.0  # 转换为0-1范围
                return min(max(score, 0.0), 1.0)
            else:
                return 0.5  # 默认中等分数
                
        except Exception as e:
            print(f"语义评分失败: {e}")
            return 0.5
    
    def simple_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """简化的重排序方法（仅使用词汇相似度）"""
        top_k = top_k or len(documents)
        
        print(f"使用简化方法重排序 {len(documents)} 个文档")
        
        scored_docs = []
        for doc in documents:
            # 计算词汇相似度
            lexical_score = self.calculate_lexical_similarity(query, doc.page_content)
            
            # 简单的长度惩罚（避免过短的文档得分过高）
            length_factor = min(len(doc.page_content) / 100, 1.0)
            
            # 综合评分
            final_score = lexical_score * 0.8 + length_factor * 0.2
            
            scored_docs.append((doc, final_score))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        result = scored_docs[:top_k]
        
        print(f"重排序完成，返回Top-{len(result)}")
        for i, (doc, score) in enumerate(result, 1):
            print(f"   {i}. 分数: {score:.3f} | 内容: {doc.page_content[:50]}...")
        
        return result
    
    def advanced_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """高级重排序方法（使用LLM语义评分）"""
        top_k = top_k or len(documents)
        
        print(f"使用高级方法重排序 {len(documents)} 个文档")
        print("注意：此方法会消耗较多API调用")
        
        scored_docs = []
        for i, doc in enumerate(documents, 1):
            print(f"   评估文档 {i}/{len(documents)}...")
            
            # 计算词汇相似度
            lexical_score = self.calculate_lexical_similarity(query, doc.page_content)
            
            # 计算语义相似度（使用LLM）
            semantic_score = self.calculate_semantic_score(query, doc.page_content)
            
            # 综合评分
            final_score = lexical_score * 0.3 + semantic_score * 0.7
            
            scored_docs.append((doc, final_score))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        result = scored_docs[:top_k]
        
        print(f"高级重排序完成，返回Top-{len(result)}")
        for i, (doc, score) in enumerate(result, 1):
            print(f"   {i}. 分数: {score:.3f} | 内容: {doc.page_content[:50]}...")
        
        return result
    
    def compare_ranking_methods(self, query: str, documents: List[Document], top_k: int = 5) -> Dict:
        """比较不同排序方法的效果"""
        print(f"比较排序方法，查询: {query}")
        
        # 原始顺序（向量相似度排序）
        original_order = [(doc, 0.0) for doc in documents[:top_k]]
        
        # 简化重排序
        simple_ranking = self.simple_rerank(query, documents, top_k)
        
        # 高级重排序（可选，消耗较多API）
        # advanced_ranking = self.advanced_rerank(query, documents, top_k)
        
        return {
            "query": query,
            "original_order": original_order,
            "simple_ranking": simple_ranking,
            # "advanced_ranking": advanced_ranking,
            "documents_count": len(documents)
        }

# 创建全局重排序器实例
reranker = SimpleReranker()
