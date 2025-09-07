import openai
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config.settings import settings
from typing import List, Tuple
import os
import numpy as np

class CompatibleEmbeddings:
    """兼容的向量化类"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """向量化多个文档 - 批处理优化"""
        embeddings = []
        batch_size = 10  # 批处理大小
        
        print(f"正在向量化 {len(texts)} 个文档块...")
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # 批量调用API
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                # 提取向量
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"批次 {i//batch_size + 1} 处理失败: {e}")
                # 回退到单个处理
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=text
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as single_e:
                        print(f"单个文档向量化失败: {single_e}")
                        # 使用零向量作为备用
                        embeddings.append([0.0] * 1536)
        
        print(f"向量化完成，共 {len(embeddings)} 个向量")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """向量化单个查询"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

class VectorStoreManager:
    """向量存储管理器 - 兼容版本"""
    
    def __init__(self):
        """初始化向量存储"""
        self.embeddings = CompatibleEmbeddings()
        
        # 确保向量数据库目录存在
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # 初始化或加载向量数据库
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量数据库"""
        print(f"正在向量化 {len(documents)} 个文档块...")
        
        # 添加文档并获取ID
        ids = self.vectorstore.add_documents(documents)
        
        # 持久化存储
        self.vectorstore.persist()
        
        print(f"向量化完成，生成 {len(ids)} 个向量")
        return ids
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """相似度搜索"""
        k = k or settings.TOP_K
        print(f"正在搜索相关文档，返回Top-{k}...")
        
        # 执行相似度搜索
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"找到 {len(results)} 个相关文档块")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """带相似度分数的搜索"""
        k = k or settings.TOP_K
        print(f"正在搜索相关文档（带分数），返回Top-{k}...")
        
        # 执行带分数的相似度搜索
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"找到 {len(results)} 个相关文档块")
        for i, (doc, score) in enumerate(results, 1):
            print(f"   {i}. 相似度: {score:.3f} | 内容: {doc.page_content[:50]}...")
        
        return results
    
    def get_stats(self) -> dict:
        """获取向量数据库统计信息"""
        try:
            # 获取向量数据库中的文档数量
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                'total_documents': count,
                'embedding_model': 'text-embedding-ada-002',
                'vector_dimension': 1536,
                'persist_directory': settings.CHROMA_PERSIST_DIR
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}

# 创建全局向量存储管理器实例
vector_store = VectorStoreManager()
