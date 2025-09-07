"""
增强版RAG问答链
集成HyDE检索和Rerank重排序
"""
import openai
from core.vector_store_compatible import vector_store
from core.hyde_retrieval import hyde_retriever
from core.reranker import reranker
from config.settings import settings
from typing import Dict, List
import time
import json

class EnhancedRAGChain:
    """增强版RAG问答链条"""
    
    def __init__(self):
        """初始化增强版RAG链条"""
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # 提示模板
        self.system_prompt = """你是一个专业的AI助手。请基于以下文档内容回答用户的问题。

请严格遵循以下要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有足够信息，请明确说明
3. 回答要简洁明了，突出重点
4. 适当引用文档中的原文
5. 用中文回答

相关文档内容：
{context}

用户问题：{question}

请提供准确、有用的回答："""
    
    def calculate_enhanced_confidence(self, source_docs, question, answer, retrieval_method="basic"):
        """增强的置信度计算"""
        if not source_docs:
            return 0.1
        
        # 基础分数
        doc_score = min(len(source_docs) / settings.TOP_K, 1.0) * 0.3
        
        # 检索方法加分
        method_bonus = 0.1 if retrieval_method == "hyde" else 0.0
        
        # 相关性分数
        relevance_score = 0
        question_words = set(question.lower().split())
        
        for doc in source_docs:
            doc_words = set(doc.page_content.lower().split())
            if question_words and doc_words:
                overlap = len(question_words & doc_words) / len(question_words | doc_words)
                relevance_score += overlap
        
        relevance_score = min(relevance_score / len(source_docs), 1.0) * 0.3 if source_docs else 0
        
        # 答案质量分数
        answer_length_score = min(len(answer.split()) / 50, 1.0) * 0.2
        
        # 关键词匹配分数
        keyword_score = 0
        for word in question_words:
            if len(word) > 2 and word in answer.lower():
                keyword_score += 1
        keyword_score = min(keyword_score / max(len(question_words), 1), 1.0) * 0.1
        
        total_confidence = doc_score + relevance_score + answer_length_score + keyword_score + method_bonus
        
        return min(max(total_confidence, 0.1), 0.95)
    
    def ask_basic(self, question: str) -> Dict:
        """基础问答方法"""
        print(f"🔵 使用基础检索方法")
        start_time = time.time()
        
        try:
            # 1. 基础检索
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
                    "chunk_index": doc.metadata.get("chunk_id", i),
                    "retrieval_method": "basic"
                })
            
            context = "\n\n".join(context_parts)
            
            # 3. 生成答案
            full_prompt = self.system_prompt.format(context=context, question=question)
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # 4. 计算置信度
            confidence = self.calculate_enhanced_confidence(source_docs, question, answer, "basic")
            
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs),
                "tokens_used": tokens_used,
                "retrieval_method": "basic"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0,
                "tokens_used": 0,
                "retrieval_method": "basic"
            }
    
    def ask_hyde(self, question: str) -> Dict:
        """使用HyDE检索的问答方法"""
        print(f"中 使用HyDE检索方法")
        start_time = time.time()
        
        try:
            # 1. HyDE检索
            source_docs = hyde_retriever.hyde_retrieve(question, k=settings.TOP_K)
            
            # 2. 构建上下文
            context_parts = []
            citations = []
            
            for i, doc in enumerate(source_docs):
                context_parts.append(f"文档片段 {i+1}:\n{doc.page_content}")
                citations.append({
                    "chunk_id": i,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_id", i),
                    "retrieval_method": "hyde"
                })
            
            context = "\n\n".join(context_parts)
            
            # 3. 生成答案
            full_prompt = self.system_prompt.format(context=context, question=question)
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # 4. 计算置信度
            confidence = self.calculate_enhanced_confidence(source_docs, question, answer, "hyde")
            
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs),
                "tokens_used": tokens_used,
                "retrieval_method": "hyde"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0,
                "tokens_used": 0,
                "retrieval_method": "hyde"
            }
    
    def ask_rerank(self, question: str) -> Dict:
        """使用重排序的问答方法"""
        print(f"🟠 使用重排序方法")
        start_time = time.time()
        
        try:
            # 1. 检索更多文档
            candidate_docs = vector_store.similarity_search(question, k=settings.TOP_K * 2)
            
            # 2. 重排序
            ranked_docs = reranker.simple_rerank(question, candidate_docs, top_k=settings.TOP_K)
            
            # 3. 提取文档
            source_docs = [doc for doc, score in ranked_docs]
            
            # 4. 构建上下文
            context_parts = []
            citations = []
            
            for i, (doc, score) in enumerate(ranked_docs):
                context_parts.append(f"文档片段 {i+1}:\n{doc.page_content}")
                citations.append({
                    "chunk_id": i,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_id", i),
                    "retrieval_method": "rerank",
                    "rerank_score": round(score, 3)
                })
            
            context = "\n\n".join(context_parts)
            
            # 5. 生成答案
            full_prompt = self.system_prompt.format(context=context, question=question)
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # 6. 计算置信度
            confidence = self.calculate_enhanced_confidence(source_docs, question, answer, "rerank")
            
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs),
                "tokens_used": tokens_used,
                "retrieval_method": "rerank"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0,
                "tokens_used": 0,
                "retrieval_method": "rerank"
            }
    
    def ask_enhanced(self, question: str) -> Dict:
        """使用HyDE+重排序的增强问答方法"""
        print(f"高 使用HyDE+重排序增强方法")
        start_time = time.time()
        
        try:
            # 1. HyDE检索获取候选文档
            candidate_docs = hyde_retriever.hyde_retrieve(question, k=settings.TOP_K * 2)
            
            # 2. 重排序
            ranked_docs = reranker.simple_rerank(question, candidate_docs, top_k=settings.TOP_K)
            
            # 3. 提取文档
            source_docs = [doc for doc, score in ranked_docs]
            
            # 4. 构建上下文
            context_parts = []
            citations = []
            
            for i, (doc, score) in enumerate(ranked_docs):
                context_parts.append(f"文档片段 {i+1}:\n{doc.page_content}")
                citations.append({
                    "chunk_id": i,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_id", i),
                    "retrieval_method": "hyde+rerank",
                    "rerank_score": round(score, 3)
                })
            
            context = "\n\n".join(context_parts)
            
            # 5. 生成答案
            full_prompt = self.system_prompt.format(context=context, question=question)
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # 6. 计算置信度
            confidence = self.calculate_enhanced_confidence(source_docs, question, answer, "hyde+rerank")
            
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs),
                "tokens_used": tokens_used,
                "retrieval_method": "hyde+rerank"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0,
                "tokens_used": 0,
                "retrieval_method": "hyde+rerank"
            }
    
    def compare_methods(self, question: str) -> Dict:
        """比较不同检索方法的效果"""
        print(f"比较不同检索方法，问题: {question}")
        
        methods = {
            "basic": self.ask_basic,
            "hyde": self.ask_hyde,
            "rerank": self.ask_rerank,
            "enhanced": self.ask_enhanced
        }
        
        results = {}
        for method_name, method_func in methods.items():
            print(f"\n--- 测试{method_name}方法 ---")
            results[method_name] = method_func(question)
        
        return {
            "question": question,
            "results": results,
            "comparison": {
                "confidence_ranking": sorted(
                    [(k, v["confidence"]) for k, v in results.items()],
                    key=lambda x: x[1], reverse=True
                ),
                "speed_ranking": sorted(
                    [(k, v["response_time"]) for k, v in results.items()],
                    key=lambda x: x[1]
                )
            }
        }

# 创建全局增强版RAG链条实例
enhanced_rag_chain = EnhancedRAGChain()
