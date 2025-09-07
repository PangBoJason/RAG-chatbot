from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import settings
import os
import hashlib
import pickle
import time
from typing import List

class DocumentLoader:
    """文档加载和处理器 - 优化版本"""
    
    def __init__(self):
        """初始化文档加载器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 创建缓存目录
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # 64KB块
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _get_cache_path(self, file_hash: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")
    
    def _load_from_cache(self, file_hash: str) -> List[Document]:
        """从缓存加载文档块"""
        cache_path = self._get_cache_path(file_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                print(f"📋 从缓存加载文档块: {len(chunks)} 个")
                return chunks
            except Exception as e:
                print(f"缓存加载失败: {e}")
        return None
    
    def _save_to_cache(self, file_hash: str, chunks: List[Document]):
        """保存文档块到缓存"""
        cache_path = self._get_cache_path(file_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
            print(f"💾 文档块已缓存: {len(chunks)} 个")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        print(f"正在加载文档: {file_path}")
        
        # 根据文件扩展名选择加载器
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")
        
        # 加载文档
        documents = loader.load()
        print(f"文档加载成功，共 {len(documents)} 页")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        print(f"正在分割文档，分块大小: {settings.CHUNK_SIZE}")
        
        start_time = time.time()
        
        # 分割文档
        chunks = self.text_splitter.split_documents(documents)
        
        # 为每个块添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'created_at': time.time()
            })
        
        processing_time = time.time() - start_time
        print(f"文档分割完成，共 {len(chunks)} 个块，耗时 {processing_time:.2f} 秒")
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Document]:
        """完整处理文档：加载 + 分割 - 优化版本带缓存"""
        print(f"\n=== 处理文档: {os.path.basename(file_path)} ===")
        
        start_time = time.time()
        
        # 检查缓存
        file_hash = self._get_file_hash(file_path)
        cached_chunks = self._load_from_cache(file_hash)
        
        if cached_chunks:
            # 使用缓存
            filename = os.path.basename(file_path)
            for chunk in cached_chunks:
                chunk.metadata['source_file'] = filename
            
            cache_time = time.time() - start_time
            print(f"⚡ 缓存命中: {filename}")
            print(f"   - 分块数量: {len(cached_chunks)}")
            print(f"   - 加载耗时: {cache_time:.2f} 秒 (加速 ~10倍)")
            return cached_chunks
        
        # 缓存未命中，重新处理
        print("🔄 缓存未命中，开始全新处理...")
        
        # 1. 加载文档
        print("步骤 1/4: 加载文档...")
        documents = self.load_document(file_path)
        
        # 2. 分割文档
        print("步骤 2/4: 分割文档...")
        chunks = self.split_documents(documents)
        
        # 3. 保存到缓存
        print("步骤 3/4: 保存缓存...")
        self._save_to_cache(file_hash, chunks)
        
        # 4. 添加文件信息到每个块
        print("步骤 4/4: 添加元数据...")
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source_file': filename,
                'file_size': file_size,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'file_hash': file_hash
            })
        
        processing_time = time.time() - start_time
        
        print(f"✅ 文档处理完成: {filename}")
        print(f"   - 文件大小: {file_size:,} 字节")
        print(f"   - 文件哈希: {file_hash[:8]}...")
        print(f"   - 原始页数: {len(documents)}")
        print(f"   - 分块数量: {len(chunks)}")
        print(f"   - 平均块大小: {sum(len(c.page_content) for c in chunks) // len(chunks)} 字符")
        print(f"   - 处理耗时: {processing_time:.2f} 秒")
        
        return chunks

# 创建全局文档加载器实例
document_loader = DocumentLoader()