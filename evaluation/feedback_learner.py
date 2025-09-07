"""
用户反馈学习模块
基于用户反馈优化RAG系统
"""
import json
import time
from typing import Dict, List, Optional, Tuple
from database.db_manager import db_manager
from database.models import UserFeedback, QALog
from sqlalchemy import desc, func
import openai
from config.settings import settings

class FeedbackLearner:
    """基于反馈的学习器"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # 反馈分析提示模板
        self.feedback_analysis_prompt = """
分析用户反馈，提取改进建议。

原问题: {question}
AI回答: {answer}
用户反馈: {feedback}
评分: {rating}/5

请分析：
1. 用户不满意的具体原因
2. 回答存在的问题（准确性、完整性、相关性）
3. 具体改进建议

请用JSON格式返回：
{{
    "issues": ["问题1", "问题2"],
    "improvement_suggestions": ["建议1", "建议2"],
    "category": "准确性/完整性/相关性/其他"
}}"""

        # 质量改进提示模板
        self.quality_improvement_prompt = """
基于反馈改进回答质量。

原问题: {question}
原回答: {original_answer}
反馈分析: {feedback_analysis}
参考上下文: {context}

请生成一个改进的回答，重点解决反馈中提到的问题："""
    
    def collect_feedback(self, qa_log_id: int, rating: int, comment: str = "", 
                        user_id: str = "anonymous") -> bool:
        """收集用户反馈"""
        try:
            feedback = UserFeedback(
                qa_log_id=qa_log_id,
                user_id=user_id,
                rating=rating,
                comment=comment,
                timestamp=time.time()
            )
            
            success = db_manager.add_user_feedback(feedback)
            
            if success:
                print(f"反馈收集成功: 评分 {rating}/5")
                
                # 如果评分较低，自动分析问题
                if rating <= 2:
                    self.analyze_negative_feedback(qa_log_id, rating, comment)
            
            return success
            
        except Exception as e:
            print(f"反馈收集失败: {e}")
            return False
    
    def analyze_negative_feedback(self, qa_log_id: int, rating: int, comment: str):
        """分析负面反馈"""
        try:
            # 获取原始问答记录
            qa_log = db_manager.get_qa_log_by_id(qa_log_id)
            if not qa_log:
                print("找不到对应的问答记录")
                return
            
            print(f"分析负面反馈 (评分: {rating}/5)")
            
            # 使用AI分析反馈
            prompt = self.feedback_analysis_prompt.format(
                question=qa_log.question,
                answer=qa_log.answer,
                feedback=comment,
                rating=rating
            )
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                analysis = json.loads(analysis_text)
                
                print("分析结果:")
                print(f"  问题类别: {analysis.get('category', '未知')}")
                print(f"  主要问题: {', '.join(analysis.get('issues', []))}")
                print(f"  改进建议: {', '.join(analysis.get('improvement_suggestions', []))}")
                
                # 保存分析结果
                self.save_feedback_analysis(qa_log_id, analysis)
                
            except json.JSONDecodeError:
                print("反馈分析结果解析失败")
                print(f"原始分析: {analysis_text}")
        
        except Exception as e:
            print(f"反馈分析失败: {e}")
    
    def save_feedback_analysis(self, qa_log_id: int, analysis: Dict):
        """保存反馈分析结果"""
        try:
            analysis_data = {
                "qa_log_id": qa_log_id,
                "analysis": analysis,
                "timestamp": time.time()
            }
            
            # 保存到文件（简化实现）
            import os
            os.makedirs("evaluation/feedback_analysis", exist_ok=True)
            
            filename = f"evaluation/feedback_analysis/analysis_{qa_log_id}_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            print(f"反馈分析已保存: {filename}")
            
        except Exception as e:
            print(f"保存反馈分析失败: {e}")
    
    def get_feedback_statistics(self) -> Dict:
        """获取反馈统计信息"""
        try:
            session = db_manager.get_session()
            
            # 基本统计
            total_feedback = session.query(UserFeedback).count()
            
            if total_feedback == 0:
                return {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "rating_distribution": {},
                    "recent_trends": []
                }
            
            # 平均评分
            avg_rating = session.query(func.avg(UserFeedback.rating)).scalar()
            
            # 评分分布
            rating_dist = {}
            for rating in range(1, 6):
                count = session.query(UserFeedback).filter(
                    UserFeedback.rating == rating
                ).count()
                rating_dist[rating] = count
            
            # 最近趋势（最近30条反馈）
            recent_feedback = session.query(UserFeedback).order_by(
                desc(UserFeedback.timestamp)
            ).limit(30).all()
            
            recent_ratings = [f.rating for f in recent_feedback]
            recent_avg = sum(recent_ratings) / len(recent_ratings) if recent_ratings else 0
            
            # 问题分类统计
            low_rating_feedback = session.query(UserFeedback).filter(
                UserFeedback.rating <= 2
            ).all()
            
            session.close()
            
            return {
                "total_feedback": total_feedback,
                "average_rating": round(avg_rating, 2),
                "rating_distribution": rating_dist,
                "recent_average": round(recent_avg, 2),
                "low_rating_count": len(low_rating_feedback),
                "satisfaction_rate": round((rating_dist.get(4, 0) + rating_dist.get(5, 0)) / total_feedback * 100, 1)
            }
            
        except Exception as e:
            print(f"获取反馈统计失败: {e}")
            return {}
    
    def generate_improvement_suggestions(self, question_type: str = None) -> List[str]:
        """基于历史反馈生成改进建议"""
        try:
            session = db_manager.get_session()
            
            # 获取低评分反馈
            low_rating_feedback = session.query(UserFeedback).filter(
                UserFeedback.rating <= 2
            ).order_by(desc(UserFeedback.timestamp)).limit(20).all()
            
            session.close()
            
            if not low_rating_feedback:
                return ["当前反馈质量良好，继续保持！"]
            
            # 分析常见问题
            common_issues = []
            improvement_suggestions = []
            
            for feedback in low_rating_feedback:
                if feedback.comment:
                    # 简单的关键词分析
                    comment_lower = feedback.comment.lower()
                    
                    if any(word in comment_lower for word in ["不准确", "错误", "不对"]):
                        common_issues.append("准确性问题")
                    
                    if any(word in comment_lower for word in ["不完整", "太简单", "不详细"]):
                        common_issues.append("完整性问题")
                    
                    if any(word in comment_lower for word in ["不相关", "答非所问"]):
                        common_issues.append("相关性问题")
                    
                    if any(word in comment_lower for word in ["太慢", "速度"]):
                        common_issues.append("响应速度问题")
            
            # 生成针对性建议
            issue_counts = {}
            for issue in common_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # 按频率排序
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            for issue, count in sorted_issues[:3]:  # 取前3个主要问题
                if issue == "准确性问题":
                    improvement_suggestions.append(f"提升回答准确性 (出现{count}次): 加强文档质量检查、优化检索算法")
                elif issue == "完整性问题":
                    improvement_suggestions.append(f"提升回答完整性 (出现{count}次): 增加上下文长度、改进回答模板")
                elif issue == "相关性问题":
                    improvement_suggestions.append(f"提升回答相关性 (出现{count}次): 优化问题理解、改进检索策略")
                elif issue == "响应速度问题":
                    improvement_suggestions.append(f"提升响应速度 (出现{count}次): 优化模型配置、增加缓存机制")
            
            if not improvement_suggestions:
                improvement_suggestions.append("反馈质量良好，建议继续监控用户满意度")
            
            return improvement_suggestions
            
        except Exception as e:
            print(f"生成改进建议失败: {e}")
            return ["建议定期检查系统性能和用户反馈"]
    
    def create_improved_answer(self, qa_log_id: int) -> Optional[str]:
        """基于反馈创建改进的回答"""
        try:
            session = db_manager.get_session()
            
            # 获取原始问答和反馈
            qa_log = session.query(QALog).filter(QALog.id == qa_log_id).first()
            feedback = session.query(UserFeedback).filter(
                UserFeedback.qa_log_id == qa_log_id
            ).first()
            
            session.close()
            
            if not qa_log or not feedback:
                print("找不到对应的问答记录或反馈")
                return None
            
            # 分析反馈
            feedback_analysis = {
                "rating": feedback.rating,
                "comment": feedback.comment,
                "issues": ["需要改进回答质量"]
            }
            
            # 获取相关上下文（简化实现）
            context = qa_log.context[:1000] if qa_log.context else "无上下文"
            
            # 生成改进的回答
            prompt = self.quality_improvement_prompt.format(
                question=qa_log.question,
                original_answer=qa_log.answer,
                feedback_analysis=json.dumps(feedback_analysis, ensure_ascii=False),
                context=context
            )
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1000
            )
            
            improved_answer = response.choices[0].message.content.strip()
            
            print("改进回答生成完成")
            print(f"原始回答: {qa_log.answer[:100]}...")
            print(f"改进回答: {improved_answer[:100]}...")
            
            return improved_answer
            
        except Exception as e:
            print(f"创建改进回答失败: {e}")
            return None
    
    def print_feedback_report(self):
        """打印反馈报告"""
        print("\n" + "="*50)
        print("用户反馈报告")
        print("="*50)
        
        stats = self.get_feedback_statistics()
        
        if stats.get("total_feedback", 0) == 0:
            print("暂无用户反馈数据")
            return
        
        print(f"总反馈数: {stats['total_feedback']}")
        print(f"平均评分: {stats['average_rating']}/5")
        print(f"最近平均: {stats['recent_average']}/5")
        print(f"满意度: {stats['satisfaction_rate']}%")
        
        print(f"\n评分分布:")
        for rating in range(5, 0, -1):
            count = stats['rating_distribution'].get(rating, 0)
            percentage = count / stats['total_feedback'] * 100 if stats['total_feedback'] > 0 else 0
            bar = "█" * int(percentage / 5)
            print(f"  {rating}星: {count:3d} ({percentage:5.1f}%) {bar}")
        
        print(f"\n改进建议:")
        suggestions = self.generate_improvement_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

# 创建全局反馈学习器实例
feedback_learner = FeedbackLearner()
