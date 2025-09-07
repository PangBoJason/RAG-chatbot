from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from config.settings import settings
from database.models import Base, Conversation, QALog, UserFeedback, DocumentMetadata
import json
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.engine = create_engine(
            settings.mysql_url,
            echo=False,  # 设为True可以看到SQL语句
            pool_size=10,  # 连接池大小
            max_overflow=20  # 最大溢出连接数
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """创建数据库表"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("数据库表创建成功！")
        except SQLAlchemyError as e:
            print(f"数据库表创建失败: {e}")
            raise
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    def create_conversation(self, user_id: str = "anonymous") -> str:
        """创建新对话会话"""
        session = self.get_session()
        try:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id
            )
            session.add(conversation)
            session.commit()
            return session_id
        except SQLAlchemyError as e:
            session.rollback()
            print(f"创建对话失败: {e}")
            raise
        finally:
            session.close()
    
    def log_qa(self, 
               session_id: str,
               question: str,
               answer: str,
               citations: List[Dict] = None,
               confidence: float = None,
               response_time: float = None,
               tokens_used: int = None,
               model_name: str = None) -> int:
        """记录问答日志"""
        session = self.get_session()
        try:
            # 获取会话ID
            conversation = session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                raise ValueError(f"会话 {session_id} 不存在")
            
            # 创建问答记录
            qa_log = QALog(
                conversation_id=conversation.id,
                question=question,
                answer=answer,
                citations=json.dumps(citations, ensure_ascii=False) if citations else None,
                confidence=confidence,
                response_time=response_time,
                tokens_used=tokens_used,
                model_name=model_name or settings.MODEL_NAME
            )
            session.add(qa_log)
            session.commit()
            return qa_log.id
        except SQLAlchemyError as e:
            session.rollback()
            print(f"记录问答失败: {e}")
            raise
        finally:
            session.close()
    
    def add_feedback(self, qa_log_id: int, feedback_type: str, comment: str = None) -> bool:
        """添加用户反馈"""
        session = self.get_session()
        try:
            feedback = UserFeedback(
                qa_log_id=qa_log_id,
                feedback_type=feedback_type,
                comment=comment
            )
            session.add(feedback)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            print(f"添加反馈失败: {e}")
            return False
        finally:
            session.close()
    
    def log_document(self, 
                     filename: str, 
                     file_path: str,
                     file_size: int,
                     chunk_count: int = None,
                     processing_time: float = None) -> int:
        """记录文档元数据"""
        session = self.get_session()
        try:
            doc_metadata = DocumentMetadata(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                chunk_count=chunk_count,
                processing_time=processing_time,
                is_processed=chunk_count is not None
            )
            session.add(doc_metadata)
            session.commit()
            return doc_metadata.id
        except SQLAlchemyError as e:
            session.rollback()
            print(f"记录文档失败: {e}")
            raise
        finally:
            session.close()
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter_by(session_id=session_id).first()
            if not conversation:
                return []
            
            qa_logs = session.query(QALog).filter_by(conversation_id=conversation.id).order_by(QALog.created_at).all()
            
            history = []
            for qa in qa_logs:
                history.append({
                    "id": qa.id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "citations": json.loads(qa.citations) if qa.citations else None,
                    "confidence": qa.confidence,
                    "created_at": qa.created_at.isoformat()
                })
            return history
        except SQLAlchemyError as e:
            print(f"获取对话历史失败: {e}")
            return []
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        session = self.get_session()
        try:
            stats = {}
            
            # 总对话数
            stats['total_conversations'] = session.query(Conversation).count()
            
            # 总问答数
            stats['total_qa'] = session.query(QALog).count()
            
            # 总文档数
            stats['total_documents'] = session.query(DocumentMetadata).count()
            
            # 用户反馈统计
            total_feedback = session.query(UserFeedback).count()
            positive_feedback = session.query(UserFeedback).filter_by(feedback_type='like').count()
            stats['feedback_rate'] = positive_feedback / total_feedback if total_feedback > 0 else 0
            
            return stats
        except SQLAlchemyError as e:
            print(f"获取统计信息失败: {e}")
            return {}
        finally:
            session.close()
    
    def get_qa_stats(self) -> Dict[str, Any]:
        """获取问答统计"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            # 总数和平均置信度
            result = session.query(
                func.count(QALog.id).label('total_count'),
                func.avg(QALog.confidence).label('avg_confidence')
            ).first()
            
            return {
                'total_count': result.total_count or 0,
                'avg_confidence': float(result.avg_confidence or 0.0)
            }
        except SQLAlchemyError as e:
            print(f"获取问答统计失败: {e}")
            return {'total_count': 0, 'avg_confidence': 0.0}
        finally:
            session.close()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            result = session.query(
                func.count(DocumentMetadata.id).label('total_count'),
                func.sum(DocumentMetadata.chunk_count).label('total_chunks'),
                func.avg(DocumentMetadata.processing_time).label('avg_processing_time')
            ).first()
            
            return {
                'total_count': result.total_count or 0,
                'total_chunks': result.total_chunks or 0,
                'avg_processing_time': float(result.avg_processing_time or 0.0)
            }
        except SQLAlchemyError as e:
            print(f"获取文档统计失败: {e}")
            return {'total_count': 0, 'total_chunks': 0, 'avg_processing_time': 0.0}
        finally:
            session.close()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            total_count = session.query(func.count(UserFeedback.id)).scalar() or 0
            positive_count = session.query(func.count(UserFeedback.id)).filter_by(feedback_type='like').scalar() or 0
            negative_count = session.query(func.count(UserFeedback.id)).filter_by(feedback_type='dislike').scalar() or 0
            
            return {
                'total_count': total_count,
                'positive_count': positive_count,
                'negative_count': negative_count
            }
        except SQLAlchemyError as e:
            print(f"获取反馈统计失败: {e}")
            return {'total_count': 0, 'positive_count': 0, 'negative_count': 0}
        finally:
            session.close()
    
    def create_document_metadata(self, filename: str, file_size: int, file_path: str = None) -> DocumentMetadata:
        """创建文档元数据记录"""
        session = self.get_session()
        try:
            doc_metadata = DocumentMetadata(
                filename=filename,
                file_path=file_path,
                file_size=file_size
            )
            session.add(doc_metadata)
            session.commit()
            # 刷新对象以确保ID被设置
            session.refresh(doc_metadata)
            # 分离对象以便在session关闭后仍可使用
            session.expunge(doc_metadata)
            return doc_metadata
        except SQLAlchemyError as e:
            session.rollback()
            print(f"创建文档元数据失败: {e}")
            raise
        finally:
            session.close()
    
    def update_document_metadata(self, doc_id: int, **kwargs):
        """更新文档元数据"""
        session = self.get_session()
        try:
            doc = session.query(DocumentMetadata).filter_by(id=doc_id).first()
            if doc:
                for key, value in kwargs.items():
                    if hasattr(doc, key):
                        setattr(doc, key, value)
                session.commit()
                print(f"✅ 更新文档元数据成功: {kwargs}")
            else:
                print(f"❌ 未找到文档ID: {doc_id}")
        except SQLAlchemyError as e:
            session.rollback()
            print(f"更新文档元数据失败: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_qa_for_evaluation(self, limit: int = 10) -> List[Dict]:
        """获取最近的问答记录用于评测"""
        session = self.get_session()
        try:
            qa_logs = session.query(QALog).order_by(QALog.created_at.desc()).limit(limit).all()
            
            qa_data = []
            for qa in qa_logs:
                qa_data.append({
                    'question': qa.question,
                    'answer': qa.answer,
                    'citations': json.loads(qa.citations) if qa.citations else [],
                    'ground_truth': qa.answer  # 这里可以后续改进为真实标准答案
                })
            
            return qa_data
        except SQLAlchemyError as e:
            print(f"获取评测数据失败: {e}")
            return []
        finally:
            session.close()
    
    def save_evaluation_results(self, eval_results: Dict) -> int:
        """保存评测结果（需要创建EvaluationResult表）"""
        # 这里先返回一个模拟ID，后续可以创建专门的评测结果表
        print(f"评测结果: {eval_results}")
        return int(time.time())
    
    def get_evaluation_history(self) -> List[Dict]:
        """获取历史评测结果"""
        # 这里先返回空列表，后续实现评测历史表
        return []
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """获取分析数据"""
        session = self.get_session()
        try:
            from sqlalchemy import func, text
            from datetime import datetime, timedelta
            
            analytics = {}
            
            # 每日问答趋势（最近7天）
            seven_days_ago = datetime.now() - timedelta(days=7)
            daily_qa = session.execute(text("""
                SELECT DATE(created_at) as date, COUNT(*) as count 
                FROM qa_logs 
                WHERE created_at >= :seven_days_ago 
                GROUP BY DATE(created_at) 
                ORDER BY date
            """), {"seven_days_ago": seven_days_ago}).fetchall()
            
            analytics['daily_qa'] = [{'date': str(row.date), 'count': row.count} for row in daily_qa]
            
            # 置信度分布
            confidence_dist = session.execute(text("""
                SELECT 
                    CASE 
                        WHEN confidence >= 0.8 THEN 'High (0.8+)'
                        WHEN confidence >= 0.6 THEN 'Medium (0.6-0.8)'
                        WHEN confidence >= 0.4 THEN 'Low (0.4-0.6)'
                        ELSE 'Very Low (<0.4)'
                    END as `range`,
                    COUNT(*) as count
                FROM qa_logs 
                WHERE confidence IS NOT NULL
                GROUP BY 
                    CASE 
                        WHEN confidence >= 0.8 THEN 'High (0.8+)'
                        WHEN confidence >= 0.6 THEN 'Medium (0.6-0.8)'
                        WHEN confidence >= 0.4 THEN 'Low (0.4-0.6)'
                        ELSE 'Very Low (<0.4)'
                    END
            """)).fetchall()
            
            analytics['confidence_distribution'] = [{'range': row.range, 'count': row.count} for row in confidence_dist]
            
            # 反馈统计
            feedback_stats = session.query(
                UserFeedback.feedback_type,
                func.count(UserFeedback.id).label('count')
            ).group_by(UserFeedback.feedback_type).all()
            
            feedback_dict = {row.feedback_type: row.count for row in feedback_stats}
            analytics['feedback_stats'] = {
                'positive': feedback_dict.get('like', 0),
                'negative': feedback_dict.get('dislike', 0)
            }
            
            return analytics
            
        except SQLAlchemyError as e:
            print(f"获取分析数据失败: {e}")
            return {}
        finally:
            session.close()

# 创建全局数据库管理器实例
db_manager = DatabaseManager()