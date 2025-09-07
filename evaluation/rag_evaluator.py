"""
RAGAS自动化评测模块
用于评估RAG系统的性能
"""
import json
import time
from typing import Dict, List, Tuple
from database.db_manager import db_manager
from core.enhanced_rag_chain import enhanced_rag_chain
import openai
from config.settings import settings

class RAGEvaluator:
    """RAG系统评测器"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # 评测指标的提示模板
        self.faithfulness_prompt = """
请评估AI回答相对于给定上下文的忠实度。

上下文: {context}
问题: {question}
AI回答: {answer}

评分标准:
- 1.0: 回答完全基于上下文，没有编造信息
- 0.8: 回答主要基于上下文，有少量合理推理
- 0.6: 回答部分基于上下文，有一些超出范围的内容
- 0.4: 回答少量基于上下文，大部分是推测
- 0.2: 回答基本不基于上下文
- 0.0: 回答完全编造，与上下文无关

请只返回数字评分(0.0-1.0):"""

        self.relevance_prompt = """
请评估AI回答对用户问题的相关性。

问题: {question}
AI回答: {answer}

评分标准:
- 1.0: 完全回答了问题，高度相关
- 0.8: 很好地回答了问题，相关性强
- 0.6: 基本回答了问题，相关性中等
- 0.4: 部分回答了问题，相关性较低
- 0.2: 几乎没有回答问题
- 0.0: 完全没有回答问题

请只返回数字评分(0.0-1.0):"""

        self.completeness_prompt = """
请评估AI回答的完整性。

问题: {question}
AI回答: {answer}

评分标准:
- 1.0: 回答非常完整详细，涵盖了问题的各个方面
- 0.8: 回答比较完整，涵盖了主要方面
- 0.6: 回答基本完整，但遗漏了一些重要信息
- 0.4: 回答不够完整，遗漏了较多信息
- 0.2: 回答很不完整，只涉及了少量内容
- 0.0: 回答极不完整或没有实质内容

请只返回数字评分(0.0-1.0):"""
    
    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> float:
        """评估回答的忠实度"""
        try:
            prompt = self.faithfulness_prompt.format(
                context=context[:1000],  # 限制上下文长度
                question=question,
                answer=answer
            )
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # 提取数字评分
            import re
            score_match = re.search(r'[0-1]\.?\d*', score_text)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0.0), 1.0)
            else:
                return 0.5  # 默认分数
                
        except Exception as e:
            print(f"忠实度评估失败: {e}")
            return 0.5
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """评估回答的相关性"""
        try:
            prompt = self.relevance_prompt.format(
                question=question,
                answer=answer
            )
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # 提取数字评分
            import re
            score_match = re.search(r'[0-1]\.?\d*', score_text)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0.0), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            print(f"相关性评估失败: {e}")
            return 0.5
    
    def evaluate_completeness(self, question: str, answer: str) -> float:
        """评估回答的完整性"""
        try:
            prompt = self.completeness_prompt.format(
                question=question,
                answer=answer
            )
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # 提取数字评分
            import re
            score_match = re.search(r'[0-1]\.?\d*', score_text)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0.0), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            print(f"完整性评估失败: {e}")
            return 0.5
    
    def evaluate_single_qa(self, question: str, answer: str, context: str) -> Dict:
        """评估单个问答对"""
        print(f"评估问答: {question[:50]}...")
        
        start_time = time.time()
        
        # 并发评估各项指标
        faithfulness = self.evaluate_faithfulness(question, answer, context)
        relevance = self.evaluate_relevance(question, answer)
        completeness = self.evaluate_completeness(question, answer)
        
        # 计算综合分数
        overall_score = (faithfulness * 0.4 + relevance * 0.4 + completeness * 0.2)
        
        evaluation_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "context": context[:200] + "..." if len(context) > 200 else context,
            "metrics": {
                "faithfulness": round(faithfulness, 3),
                "relevance": round(relevance, 3),
                "completeness": round(completeness, 3),
                "overall_score": round(overall_score, 3)
            },
            "evaluation_time": round(evaluation_time, 2)
        }
        
        print(f"   忠实度: {faithfulness:.3f}")
        print(f"   相关性: {relevance:.3f}")
        print(f"   完整性: {completeness:.3f}")
        print(f"   综合分: {overall_score:.3f}")
        
        return result
    
    def create_evaluation_dataset(self) -> List[Dict]:
        """创建评测数据集"""
        evaluation_dataset = [
            {
                "question": "什么是RAG技术？",
                "expected_topics": ["检索增强生成", "文档检索", "LLM", "向量数据库"]
            },
            {
                "question": "RAG有什么主要优势？",
                "expected_topics": ["实时性", "准确性", "可解释性", "灵活性"]
            },
            {
                "question": "RAG的工作流程是什么？",
                "expected_topics": ["文档预处理", "向量化", "检索", "生成答案"]
            },
            {
                "question": "向量数据库在RAG中起什么作用？",
                "expected_topics": ["语义搜索", "向量存储", "相似度计算"]
            },
            {
                "question": "如何提高RAG系统的准确性？",
                "expected_topics": ["文档质量", "分块策略", "检索优化", "重排序"]
            }
        ]
        
        return evaluation_dataset
    
    def run_method_comparison(self, dataset: List[Dict]) -> Dict:
        """运行不同方法的对比评测"""
        print("运行方法对比评测...")
        
        methods = {
            "basic": enhanced_rag_chain.ask_basic,
            "hyde": enhanced_rag_chain.ask_hyde,
            "rerank": enhanced_rag_chain.ask_rerank,
            "enhanced": enhanced_rag_chain.ask_enhanced
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"\n--- 评测 {method_name.upper()} 方法 ---")
            
            method_results = []
            
            for i, test_case in enumerate(dataset, 1):
                question = test_case["question"]
                print(f"\n{i}. 问题: {question}")
                
                # 获取答案
                qa_result = method_func(question)
                
                # 构建上下文
                context = ""
                if qa_result["citations"]:
                    context = "\n".join([c["content"] for c in qa_result["citations"]])
                
                # 评测
                evaluation = self.evaluate_single_qa(
                    question=question,
                    answer=qa_result["answer"],
                    context=context
                )
                
                # 添加方法信息
                evaluation["method"] = method_name
                evaluation["response_time"] = qa_result["response_time"]
                evaluation["confidence"] = qa_result["confidence"]
                evaluation["tokens_used"] = qa_result.get("tokens_used", 0)
                
                method_results.append(evaluation)
            
            results[method_name] = method_results
        
        return results
    
    def generate_evaluation_report(self, results: Dict) -> Dict:
        """生成评测报告"""
        print("\n生成评测报告...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methods_compared": list(results.keys()),
            "total_questions": len(list(results.values())[0]) if results else 0,
            "method_performance": {},
            "overall_ranking": []
        }
        
        # 计算各方法的平均指标
        for method_name, method_results in results.items():
            if not method_results:
                continue
            
            metrics = {
                "avg_faithfulness": sum(r["metrics"]["faithfulness"] for r in method_results) / len(method_results),
                "avg_relevance": sum(r["metrics"]["relevance"] for r in method_results) / len(method_results),
                "avg_completeness": sum(r["metrics"]["completeness"] for r in method_results) / len(method_results),
                "avg_overall_score": sum(r["metrics"]["overall_score"] for r in method_results) / len(method_results),
                "avg_response_time": sum(r["response_time"] for r in method_results) / len(method_results),
                "avg_confidence": sum(r["confidence"] for r in method_results) / len(method_results),
                "total_tokens": sum(r.get("tokens_used", 0) for r in method_results)
            }
            
            report["method_performance"][method_name] = {
                "metrics": {k: round(v, 3) for k, v in metrics.items()},
                "detailed_results": method_results
            }
        
        # 生成排名
        method_scores = []
        for method, perf in report["method_performance"].items():
            overall_score = perf["metrics"]["avg_overall_score"]
            method_scores.append((method, overall_score))
        
        method_scores.sort(key=lambda x: x[1], reverse=True)
        report["overall_ranking"] = method_scores
        
        return report
    
    def save_evaluation_results(self, report: Dict, filename: str = None):
        """保存评测结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        try:
            import os
            os.makedirs("evaluation", exist_ok=True)
            
            filepath = os.path.join("evaluation", filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"评测报告已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存评测报告失败: {e}")
            return None
    
    def print_summary_report(self, report: Dict):
        """打印评测摘要报告"""
        print("\n" + "="*60)
        print("RAG系统评测报告摘要")
        print("="*60)
        
        print(f"评测时间: {report['timestamp']}")
        print(f"测试问题数: {report['total_questions']}")
        print(f"对比方法数: {len(report['methods_compared'])}")
        
        print(f"\n方法排名（按综合得分）:")
        for rank, (method, score) in enumerate(report['overall_ranking'], 1):
            print(f"   {rank}. {method.upper()}: {score:.3f}")
        
        print(f"\n详细指标对比:")
        for method, perf in report['method_performance'].items():
            metrics = perf['metrics']
            print(f"\n{method.upper()} 方法:")
            print(f"   忠实度: {metrics['avg_faithfulness']:.3f}")
            print(f"   相关性: {metrics['avg_relevance']:.3f}")
            print(f"   完整性: {metrics['avg_completeness']:.3f}")
            print(f"   综合分: {metrics['avg_overall_score']:.3f}")
            print(f"   平均用时: {metrics['avg_response_time']:.2f}秒")
            print(f"   平均置信度: {metrics['avg_confidence']:.3f}")
            print(f"   总Token消耗: {metrics['total_tokens']}")
        
        # 推荐最佳方法
        if report['overall_ranking']:
            best_method = report['overall_ranking'][0][0]
            best_score = report['overall_ranking'][0][1]
            
            print(f"\n推荐:")
            print(f"   最佳方法: {best_method.upper()}")
            print(f"   综合得分: {best_score:.3f}")
            
            best_perf = report['method_performance'][best_method]['metrics']
            print(f"   该方法在忠实度、相关性和完整性方面表现优异")

# 创建全局评测器实例
rag_evaluator = RAGEvaluator()
