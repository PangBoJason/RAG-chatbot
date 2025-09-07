"""
智能Agent核心模块
实现工具调用、决策制定和任务执行
"""
import openai
from typing import Dict, List, Any, Optional
from config.settings import settings
import json
import re

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        
        # 知识库问答工具
        self.register_tool(
            name="knowledge_base_search",
            description="在本地知识库中搜索相关信息回答问题",
            parameters={
                "query": {"type": "string", "description": "用户的问题"}
            },
            function=self._knowledge_base_search
        )
        
        # 计算器工具
        self.register_tool(
            name="calculator",
            description="执行数学计算",
            parameters={
                "expression": {"type": "string", "description": "数学表达式"}
            },
            function=self._calculator
        )
        
        # 搜索引擎工具（模拟）
        self.register_tool(
            name="search_engine",
            description="在互联网上搜索最新信息",
            parameters={
                "query": {"type": "string", "description": "搜索关键词"}
            },
            function=self._search_engine
        )
        
        # 文件分析工具
        self.register_tool(
            name="file_analyzer",
            description="分析上传的文档内容",
            parameters={
                "file_content": {"type": "string", "description": "文件内容"},
                "analysis_type": {"type": "string", "description": "分析类型"}
            },
            function=self._file_analyzer
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict, function):
        """注册工具"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            },
            "function": function
        }
    
    def get_tool_definitions(self, selected_tools: List[str] = None) -> List[Dict]:
        """获取工具定义"""
        if selected_tools:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info["description"],
                        "parameters": tool_info["parameters"]
                    }
                }
                for tool_name, tool_info in self.tools.items()
                if tool_name in selected_tools
            ]
        else:
            return [
                {
                    "type": "function", 
                    "function": {
                        "name": tool_name,
                        "description": tool_info["description"],
                        "parameters": tool_info["parameters"]
                    }
                }
                for tool_name, tool_info in self.tools.items()
            ]
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """执行工具"""
        if tool_name not in self.tools:
            return {"error": f"未知工具: {tool_name}"}
        
        try:
            result = self.tools[tool_name]["function"](**parameters)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": f"工具执行失败: {str(e)}"}
    
    def _knowledge_base_search(self, query: str) -> Dict[str, Any]:
        """知识库搜索实现"""
        try:
            from core.enhanced_rag_chain import enhanced_rag_chain
            result = enhanced_rag_chain.ask_enhanced(query)
            return result
        except Exception as e:
            return {
                "answer": f"知识库搜索失败: {str(e)}",
                "confidence": 0.0,
                "citations": []
            }
    
    def _calculator(self, expression: str) -> Dict[str, Any]:
        """计算器实现"""
        try:
            # 安全的数学计算
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {
                    "calculation": expression,
                    "result": result,
                    "success": True
                }
            else:
                return {"error": "不安全的表达式"}
        except Exception as e:
            return {"error": f"计算失败: {str(e)}"}
    
    def _search_engine(self, query: str) -> Dict[str, Any]:
        """搜索引擎实现（模拟）"""
        return {
            "query": query,
            "results": [
                {"title": "搜索结果1", "url": "http://example.com", "snippet": "这是模拟的搜索结果"},
                {"title": "搜索结果2", "url": "http://example.com", "snippet": "另一个模拟结果"}
            ],
            "note": "这是模拟的搜索结果，实际使用时需要集成真实的搜索API"
        }
    
    def _file_analyzer(self, file_content: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """文件分析实现"""
        return {
            "content_length": len(file_content),
            "analysis_type": analysis_type,
            "summary": f"文档包含{len(file_content)}个字符，类型为{analysis_type}分析",
            "note": "这是简化的文件分析实现"
        }

class IntelligentAgent:
    """智能Agent"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
    
    def process_request(self, user_input: str, selected_tools: List[str] = None, 
                       agent_enabled: bool = True) -> Dict[str, Any]:
        """处理用户请求"""
        try:
            if agent_enabled and selected_tools and len(selected_tools) > 1:
                # 使用Agent模式进行工具选择和调用
                return self._agent_mode_process(user_input, selected_tools)
            elif selected_tools and len(selected_tools) == 1:
                # 直接使用指定工具
                return self._direct_tool_process(user_input, selected_tools[0])
            else:
                # 普通对话模式
                return self._normal_chat_process(user_input)
        
        except Exception as e:
            return {
                "answer": f"处理请求时出现错误: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "tool_calls": []
            }
    
    def _agent_mode_process(self, user_input: str, selected_tools: List[str]) -> Dict[str, Any]:
        """Agent模式处理"""
        try:
            # 准备工具定义
            tool_definitions = self.tool_registry.get_tool_definitions(selected_tools)
            
            # 构建系统提示
            system_prompt = f"""你是一个智能助手Agent，可以使用以下工具来回答用户问题：

可用工具：
{self._format_tool_descriptions(selected_tools)}

请根据用户问题选择最合适的工具，并提供有用的回答。
如果需要使用工具，请调用相应的function。
如果不需要工具，可以直接回答。
"""

            # 调用OpenAI API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=0.7
            )
            
            # 处理响应
            message = response.choices[0].message
            
            if message.tool_calls:
                # 执行工具调用
                tool_results = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    result = self.tool_registry.execute_tool(tool_name, tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result
                    })
                
                # 生成最终回答
                return self._generate_final_answer(user_input, tool_results)
            
            else:
                # 直接回答
                return {
                    "answer": message.content,
                    "confidence": 0.8,
                    "citations": [],
                    "tool_calls": []
                }
        
        except Exception as e:
            return {
                "answer": f"Agent处理失败: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "tool_calls": []
            }
    
    def _direct_tool_process(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """直接工具处理"""
        # 映射工具名称
        tool_mapping = {
            "knowledge_base": "knowledge_base_search",
            "calculator": "calculator",
            "search_engine": "search_engine",
            "file_analysis": "file_analyzer"
        }
        
        actual_tool_name = tool_mapping.get(tool_name, tool_name)
        
        # 准备参数
        if actual_tool_name == "knowledge_base_search":
            parameters = {"query": user_input}
        elif actual_tool_name == "calculator":
            # 提取数学表达式
            expression = self._extract_math_expression(user_input)
            parameters = {"expression": expression}
        else:
            parameters = {"query": user_input}
        
        # 执行工具
        result = self.tool_registry.execute_tool(actual_tool_name, parameters)
        
        if result.get("success"):
            tool_result = result["result"]
            
            if actual_tool_name == "knowledge_base_search":
                return tool_result
            elif actual_tool_name == "calculator":
                calc_result = tool_result
                return {
                    "answer": f"计算结果：{calc_result.get('calculation', '')} = {calc_result.get('result', '')}",
                    "confidence": 0.95,
                    "citations": [{"content": f"数学计算: {calc_result.get('calculation', '')}", "source": "计算器"}],
                    "tool_calls": [{"tool": "calculator", "result": calc_result}]
                }
            else:
                return {
                    "answer": f"工具执行结果：{json.dumps(tool_result, ensure_ascii=False)}",
                    "confidence": 0.7,
                    "citations": [],
                    "tool_calls": [{"tool": actual_tool_name, "result": tool_result}]
                }
        else:
            return {
                "answer": f"工具执行失败：{result.get('error', '未知错误')}",
                "confidence": 0.0,
                "citations": [],
                "tool_calls": []
            }
    
    def _normal_chat_process(self, user_input: str) -> Dict[str, Any]:
        """普通聊天处理"""
        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，请尽力回答用户的问题。"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7
            )
            
            return {
                "answer": response.choices[0].message.content,
                "confidence": 0.7,
                "citations": [],
                "tool_calls": []
            }
        
        except Exception as e:
            return {
                "answer": f"普通对话处理失败: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "tool_calls": []
            }
    
    def _format_tool_descriptions(self, selected_tools: List[str]) -> str:
        """格式化工具描述"""
        tool_mapping = {
            "knowledge_base": "knowledge_base_search",
            "calculator": "calculator", 
            "search_engine": "search_engine",
            "file_analysis": "file_analyzer"
        }
        
        descriptions = []
        for tool in selected_tools:
            actual_tool = tool_mapping.get(tool, tool)
            if actual_tool in self.tool_registry.tools:
                tool_info = self.tool_registry.tools[actual_tool]
                descriptions.append(f"- {tool_info['name']}: {tool_info['description']}")
        
        return "\n".join(descriptions)
    
    def _extract_math_expression(self, text: str) -> str:
        """提取数学表达式"""
        # 简单的数学表达式提取
        import re
        pattern = r'[\d+\-*/().]+' 
        matches = re.findall(pattern, text)
        return matches[0] if matches else text
    
    def _generate_final_answer(self, user_input: str, tool_results: List[Dict]) -> Dict[str, Any]:
        """生成最终答案"""
        # 整合工具结果
        answer_parts = []
        citations = []
        
        for tool_result in tool_results:
            if tool_result["result"].get("success"):
                result_data = tool_result["result"]["result"]
                
                if tool_result["tool"] == "knowledge_base_search":
                    answer_parts.append(result_data.get("answer", ""))
                    citations.extend(result_data.get("citations", []))
                elif tool_result["tool"] == "calculator":
                    calc = result_data.get("calculation", "")
                    res = result_data.get("result", "")
                    answer_parts.append(f"计算结果：{calc} = {res}")
                    citations.append({"content": f"数学计算: {calc}", "source": "计算器"})
        
        final_answer = "\n\n".join(answer_parts) if answer_parts else "抱歉，无法处理您的请求。"
        
        # 计算综合置信度
        confidence = 0.8 if answer_parts else 0.2
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "citations": citations,
            "tool_calls": tool_results
        }

# 全局Agent实例
intelligent_agent = IntelligentAgent()
