"""
RAG评估器 - 简化版本
实现基本的忠实度、相关性、召回率评测
"""
import openai
from typing import List, Dict, Any
from config.settings import settings
import json
import re
from datetime import datetime

class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
    
    def evaluate_faithfulness(self, answer: str, citations: List[Dict]) -> float:
        """评估答案忠实度 - 答案是否基于给定的上下文"""
        if not citations:
            return 0.0
        
        context = "\n".join([c.get('content', '') for c in citations])
        
        prompt = f"""
        请评估以下答案是否忠实于给定的上下文信息。
        
        上下文:
        {context}
        
        答案:
        {answer}
        
        评估标准:
        - 1.0: 答案完全基于上下文，没有虚假信息
        - 0.8: 答案大部分基于上下文，有少量推理
        - 0.6: 答案部分基于上下文，有一些推理和扩展
        - 0.4: 答案与上下文相关但有明显扩展
        - 0.2: 答案与上下文关系较弱
        - 0.0: 答案与上下文无关或包含虚假信息
        
        请只返回一个0-1之间的数字，不要其他解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'[0-1]\.\d+|[0-1]', score_text)[0])
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"忠实度评估失败: {e}")
            return 0.5  # 默认中等分数
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性 - 答案是否直接回答了问题"""
        prompt = f"""
        请评估以下答案对问题的相关性。
        
        问题:
        {question}
        
        答案:
        {answer}
        
        评估标准:
        - 1.0: 答案完全且直接回答了问题
        - 0.8: 答案基本回答了问题，有少量偏离
        - 0.6: 答案部分回答了问题
        - 0.4: 答案与问题相关但不够直接
        - 0.2: 答案与问题关系较弱
        - 0.0: 答案与问题无关
        
        请只返回一个0-1之间的数字，不要其他解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'[0-1]\.\d+|[0-1]', score_text)[0])
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"相关性评估失败: {e}")
            return 0.5
    
    def evaluate_context_recall(self, question: str, citations: List[Dict], ground_truth: str = None) -> float:
        """评估上下文召回率 - 检索到的上下文是否包含回答问题所需的信息"""
        if not citations:
            return 0.0
        
        context = "\n".join([c.get('content', '') for c in citations])
        
        prompt = f"""
        请评估检索到的上下文是否包含回答以下问题所需的足够信息。
        
        问题:
        {question}
        
        检索到的上下文:
        {context}
        
        评估标准:
        - 1.0: 上下文包含完全回答问题所需的所有信息
        - 0.8: 上下文包含大部分必要信息
        - 0.6: 上下文包含部分必要信息
        - 0.4: 上下文包含少量相关信息
        - 0.2: 上下文信息与问题相关但不足以回答
        - 0.0: 上下文不包含回答问题所需的信息
        
        请只返回一个0-1之间的数字，不要其他解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'[0-1]\.\d+|[0-1]', score_text)[0])
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"召回率评估失败: {e}")
            return 0.5
    
    def evaluate_single_qa(self, qa_data: Dict) -> Dict[str, float]:
        """评估单个问答对"""
        question = qa_data['question']
        answer = qa_data['answer']
        citations = qa_data.get('citations', [])
        ground_truth = qa_data.get('ground_truth')
        
        # 执行各项评估
        faithfulness = self.evaluate_faithfulness(answer, citations)
        answer_relevancy = self.evaluate_answer_relevancy(question, answer)
        context_recall = self.evaluate_context_recall(question, citations, ground_truth)
        
        return {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_recall': context_recall
        }
    
    def evaluate_qa_batch(self, qa_batch: List[Dict]) -> Dict[str, float]:
        """批量评估问答对"""
        if not qa_batch:
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0,
                'count': 0
            }
        
        total_scores = {
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'context_recall': 0.0
        }
        
        valid_count = 0
        
        for qa_data in qa_batch:
            try:
                scores = self.evaluate_single_qa(qa_data)
                for metric, score in scores.items():
                    total_scores[metric] += score
                valid_count += 1
                
            except Exception as e:
                print(f"评估单个QA失败: {e}")
                continue
        
        if valid_count == 0:
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0,
                'count': 0
            }
        
        # 计算平均分
        avg_scores = {
            metric: score / valid_count 
            for metric, score in total_scores.items()
        }
        avg_scores['count'] = valid_count
        
        return avg_scores
    
    def generate_evaluation_report(self, eval_results: Dict) -> str:
        """生成评估报告"""
        report = f"""
# RAG系统评估报告

## 评估概览
- 评估样本数: {eval_results.get('count', 0)}
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估指标

### 忠实度 (Faithfulness): {eval_results.get('faithfulness', 0):.3f}
答案是否忠实于给定的上下文信息，避免幻觉和虚假信息。

### 答案相关性 (Answer Relevancy): {eval_results.get('answer_relevancy', 0):.3f}
答案是否直接且准确地回答了用户的问题。

### 上下文召回率 (Context Recall): {eval_results.get('context_recall', 0):.3f}
检索到的上下文是否包含回答问题所需的足够信息。

## 评估结论

{'🟢 优秀' if all(score >= 0.8 for score in [eval_results.get('faithfulness', 0), eval_results.get('answer_relevancy', 0), eval_results.get('context_recall', 0)]) else '🟡 良好' if all(score >= 0.6 for score in [eval_results.get('faithfulness', 0), eval_results.get('answer_relevancy', 0), eval_results.get('context_recall', 0)]) else '🔴 需要改进'}

## 改进建议
"""
        
        # 根据具体分数给出建议
        if eval_results.get('faithfulness', 0) < 0.7:
            report += "- 加强上下文约束，减少模型的自由发挥\n"
        
        if eval_results.get('answer_relevancy', 0) < 0.7:
            report += "- 优化提示词，让模型更专注于直接回答问题\n"
        
        if eval_results.get('context_recall', 0) < 0.7:
            report += "- 改进检索策略，提高相关上下文的召回率\n"
        
        return report

# 创建全局评估器实例
rag_evaluator = RAGEvaluator()
