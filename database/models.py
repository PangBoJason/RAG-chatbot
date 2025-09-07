from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    """对话会话表"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(50), default='anonymous')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联问答记录
    qa_logs = relationship("QALog", back_populates="conversation")

class QALog(Base):
    """问答记录表"""
    __tablename__ = 'qa_logs'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(Text)  # JSON格式存储引用片段
    confidence = Column(Float)  # 答案置信度
    response_time = Column(Float)  # 响应时间(秒)
    tokens_used = Column(Integer)  # 使用的token数量
    model_name = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联关系
    conversation = relationship("Conversation", back_populates="qa_logs")
    feedbacks = relationship("UserFeedback", back_populates="qa_log")

class UserFeedback(Base):
    """用户反馈表"""
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    qa_log_id = Column(Integer, ForeignKey('qa_logs.id'))
    feedback_type = Column(String(20))  # 'like', 'dislike'
    comment = Column(Text)  # 用户评论
    created_at = Column(DateTime, default=datetime.utcnow)
    
    qa_log = relationship("QALog", back_populates="feedbacks")

class DocumentMetadata(Base):
    """文档元数据表"""
    __tablename__ = 'document_metadata'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)  # 文件大小(字节)
    chunk_count = Column(Integer)  # 分块数量
    upload_time = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)
    processing_time = Column(Float)  # 处理时间(秒)