"""
多模型支持模块
支持多种LLM模型的RAG系统
"""
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import openai
from config.settings import settings

class LLMProvider(ABC):
    """LLM提供者抽象基类"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict], **kwargs) -> Dict:
        """生成回答"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI模型提供者"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=base_url or settings.OPENAI_API_BASE
        )
        self.embedding_model = "text-embedding-ada-002"
    
    def generate_response(self, messages: List[Dict], **kwargs) -> Dict:
        """生成回答"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0)
            )
            
            response_time = time.time() - start_time
            
            return {
                "content": response.choices[0].message.content,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens,
                "model": self.model_name,
                "provider": "OpenAI"
            }
            
        except Exception as e:
            return {
                "content": f"生成回答失败: {str(e)}",
                "response_time": 0,
                "tokens_used": 0,
                "model": self.model_name,
                "provider": "OpenAI",
                "error": str(e)
            }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "max_tokens": 4096 if "gpt-3.5" in self.model_name else 8192,
            "supports_function_calling": True,
            "supports_streaming": True
        }

class QwenProvider(LLMProvider):
    """通义千问模型提供者（示例实现）"""
    
    def __init__(self, model_name: str = "qwen-turbo", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        # 注意：这里需要实际的千问API客户端
        # self.client = DashScopeClient(api_key=api_key)
    
    def generate_response(self, messages: List[Dict], **kwargs) -> Dict:
        """生成回答（示例实现）"""
        try:
            # 这里应该调用实际的千问API
            # 目前返回模拟结果
            return {
                "content": "这是千问模型的模拟回答。实际使用时需要集成真实的千问API。",
                "response_time": 1.0,
                "tokens_used": 100,
                "model": self.model_name,
                "provider": "Qwen"
            }
        except Exception as e:
            return {
                "content": f"千问模型生成回答失败: {str(e)}",
                "response_time": 0,
                "tokens_used": 0,
                "model": self.model_name,
                "provider": "Qwen",
                "error": str(e)
            }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入（示例实现）"""
        # 实际实现需要调用千问的嵌入API
        return []
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "provider": "Qwen",
            "model_name": self.model_name,
            "embedding_model": "text-embedding-v1",
            "max_tokens": 6000,
            "supports_function_calling": False,
            "supports_streaming": True
        }

class ChatGLMProvider(LLMProvider):
    """ChatGLM模型提供者（示例实现）"""
    
    def __init__(self, model_name: str = "chatglm3-6b", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate_response(self, messages: List[Dict], **kwargs) -> Dict:
        """生成回答（示例实现）"""
        return {
            "content": "这是ChatGLM模型的模拟回答。实际使用时需要集成真实的ChatGLM API。",
            "response_time": 1.2,
            "tokens_used": 120,
            "model": self.model_name,
            "provider": "ChatGLM"
        }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入（示例实现）"""
        return []
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "provider": "ChatGLM",
            "model_name": self.model_name,
            "embedding_model": "text2vec-base",
            "max_tokens": 4096,
            "supports_function_calling": False,
            "supports_streaming": True
        }

class MultiModelManager:
    """多模型管理器"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化模型提供者"""
        try:
            # OpenAI GPT-3.5
            self.providers["gpt-3.5-turbo"] = OpenAIProvider("gpt-3.5-turbo")
            
            # OpenAI GPT-4（如果有API权限）
            self.providers["gpt-4"] = OpenAIProvider("gpt-4")
            
            # 其他模型（示例）
            self.providers["qwen-turbo"] = QwenProvider("qwen-turbo")
            self.providers["chatglm3-6b"] = ChatGLMProvider("chatglm3-6b")
            
            # 设置默认模型
            self.default_provider = self.providers.get("gpt-3.5-turbo")
            
            print(f"多模型管理器初始化完成，可用模型: {list(self.providers.keys())}")
            
        except Exception as e:
            print(f"多模型管理器初始化失败: {e}")
    
    def get_provider(self, model_name: str) -> Optional[LLMProvider]:
        """获取指定模型的提供者"""
        return self.providers.get(model_name, self.default_provider)
    
    def list_available_models(self) -> List[str]:
        """列出可用模型"""
        return list(self.providers.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """获取模型信息"""
        provider = self.get_provider(model_name)
        if provider:
            return provider.get_model_info()
        return {}
    
    def generate_response_with_model(self, model_name: str, messages: List[Dict], **kwargs) -> Dict:
        """使用指定模型生成回答"""
        provider = self.get_provider(model_name)
        if not provider:
            return {
                "content": f"模型 {model_name} 不可用",
                "response_time": 0,
                "tokens_used": 0,
                "model": model_name,
                "provider": "Unknown",
                "error": "Model not found"
            }
        
        return provider.generate_response(messages, **kwargs)
    
    def compare_models(self, question: str, models: List[str] = None) -> Dict:
        """比较多个模型的回答"""
        if models is None:
            models = ["gpt-3.5-turbo", "gpt-4"]  # 默认比较的模型
        
        print(f"比较模型回答: {question}")
        
        messages = [{"role": "user", "content": question}]
        results = {}
        
        for model_name in models:
            if model_name in self.providers:
                print(f"  正在获取 {model_name} 的回答...")
                
                result = self.generate_response_with_model(model_name, messages)
                results[model_name] = result
                
                if "error" not in result:
                    print(f"    {model_name}: {result['content'][:50]}...")
                else:
                    print(f"    {model_name}: {result['error']}")
            else:
                print(f"    {model_name}: 模型不可用")
        
        return results
    
    def benchmark_models(self, test_questions: List[str] = None) -> Dict:
        """模型性能基准测试"""
        if test_questions is None:
            test_questions = [
                "什么是人工智能？",
                "请解释机器学习的基本概念",
                "RAG技术有什么优势？"
            ]
        
        print("开始模型性能基准测试...")
        
        benchmark_results = {}
        
        for model_name in self.providers.keys():
            print(f"\n测试模型: {model_name}")
            
            model_results = {
                "total_questions": len(test_questions),
                "responses": [],
                "avg_response_time": 0,
                "total_tokens": 0,
                "success_rate": 0
            }
            
            successful_responses = 0
            total_time = 0
            total_tokens = 0
            
            for i, question in enumerate(test_questions, 1):
                print(f"  问题 {i}/{len(test_questions)}: {question[:30]}...")
                
                messages = [{"role": "user", "content": question}]
                result = self.generate_response_with_model(model_name, messages)
                
                model_results["responses"].append({
                    "question": question,
                    "result": result
                })
                
                if "error" not in result:
                    successful_responses += 1
                    total_time += result["response_time"]
                    total_tokens += result["tokens_used"]
                    print(f"    成功 ({result['response_time']:.2f}s)")
                else:
                    print(f"    失败: {result['error']}")
            
            model_results["success_rate"] = successful_responses / len(test_questions) * 100
            model_results["avg_response_time"] = total_time / successful_responses if successful_responses > 0 else 0
            model_results["total_tokens"] = total_tokens
            
            benchmark_results[model_name] = model_results
        
        return benchmark_results
    
    def print_benchmark_report(self, benchmark_results: Dict):
        """打印基准测试报告"""
        print("\n" + "="*60)
        print("多模型性能基准测试报告")
        print("="*60)
        
        for model_name, results in benchmark_results.items():
            print(f"\n{model_name}:")
            print(f"   成功率: {results['success_rate']:.1f}%")
            print(f"   平均响应时间: {results['avg_response_time']:.2f}秒")
            print(f"   总Token消耗: {results['total_tokens']}")
        
        # 性能排名
        print(f"\n性能排名:")
        
        # 按成功率排序
        success_ranking = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        print(f"  按成功率:")
        for rank, (model, results) in enumerate(success_ranking, 1):
            print(f"    {rank}. {model}: {results['success_rate']:.1f}%")
        
        # 按响应时间排序（成功的模型）
        speed_ranking = sorted(
            [(m, r) for m, r in benchmark_results.items() if r['success_rate'] > 0],
            key=lambda x: x[1]['avg_response_time']
        )
        
        if speed_ranking:
            print(f"  按响应速度:")
            for rank, (model, results) in enumerate(speed_ranking, 1):
                print(f"    {rank}. {model}: {results['avg_response_time']:.2f}秒")

# 创建全局多模型管理器实例
multi_model_manager = MultiModelManager()
