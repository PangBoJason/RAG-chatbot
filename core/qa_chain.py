from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from core.vector_store_compatible import vector_store
from config.settings import settings
from typing import Dict, List
import time
import json

class RAGChain:
    """RAG问答链条"""
    
    def __init__(self):
        """初始化RAG链条"""
        # 初始化语言模型
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            model_name=settings.MODEL_NAME,
            temperature=0.1  # 较低的温度确保回答更准确
        )
        
        # 创建检索器
        self.retriever = vector_store.vectorstore.as_retriever(
            search_kwargs={"k": settings.TOP_K}
        )
        
        # 定义提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是一个专业的AI助手。请基于以下文档内容回答用户的问题。

相关文档：
{context}

用户问题：{question}

请遵循以下要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁明了，突出重点
4. 可以适当引用文档中的原文

回答：
            """
        )
        
        # 创建检索问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def ask(self, question: str) -> Dict:
        """提问并获取答案"""
        print(f" 用户问题: {question}")
        
        start_time = time.time()
        
        try:
            # 执行问答
            result = self.qa_chain({"query": question})
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 提取答案和来源文档
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # 构建引用信息
            citations = []
            for i, doc in enumerate(source_docs):
                citations.append({
                    "chunk_id": i,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_id", i)
                })
            
            # 简单的置信度计算（基于检索到的文档数量和相关性）
            confidence = min(0.9, len(source_docs) / settings.TOP_K * 0.8 + 0.1)
            
            result_dict = {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "response_time": response_time,
                "source_count": len(source_docs)
            }
            
            print(f" 回答生成完成，耗时 {response_time:.2f} 秒")
            print(f" 使用了 {len(source_docs)} 个文档片段")
            
            return result_dict
            
        except Exception as e:
            print(f" 问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误：{str(e)}",
                "citations": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "source_count": 0
            }

# 创建全局RAG链条实例
rag_chain = RAGChain()