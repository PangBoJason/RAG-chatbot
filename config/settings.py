import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings:
    """项目配置类"""
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    # MySQL配置
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "raglite")
    
    # 向量数据库配置
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # RAG配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))
    TOP_K = int(os.getenv("TOP_K", 5))
    
    @property
    def mysql_url(self):
        """生成MySQL连接字符串"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"

# 创建全局配置实例
settings = Settings()