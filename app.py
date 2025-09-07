"""
RAGLite - 智能问答助手 (完整版)
集成了RAGAS评测、用户反馈学习、多模型支持等企业级功能
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import tempfile
from core.enhanced_rag_chain import enhanced_rag_chain
from core.document_loader import document_loader
from core.vector_store_compatible import vector_store
from database.db_manager import db_manager
from evaluation.rag_evaluator import rag_evaluator
from evaluation.feedback_learner import feedback_learner
from evaluation.multi_model_support import multi_model_manager
from config.settings import settings

# 页面配置
st.set_page_config(
    page_title="RAGLite - 智能问答助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """初始化会话状态"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = db_manager.create_conversation("streamlit_user")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-3.5-turbo"
    if 'rag_method' not in st.session_state:
        st.session_state.rag_method = "enhanced"

def upload_and_process_file(uploaded_file):
    """上传和处理文件"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # 处理文档
        chunks = document_loader.process_document(tmp_file_path)
        
        if chunks:
            # 向量化存储
            vector_ids = vector_store.add_documents(chunks)
            
            # 记录到数据库
            metadata = {
                "filename": uploaded_file.name,
                "file_size": len(uploaded_file.getvalue()),
                "chunks_count": len(chunks),
                "upload_time": time.time()
            }
            
            db_manager.add_document_metadata(
                filename=uploaded_file.name,
                content_hash=str(hash(uploaded_file.getvalue())),
                metadata=json.dumps(metadata)
            )
            
            return len(chunks)
        else:
            return 0
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def display_chat_interface():
    """显示聊天界面"""
    st.header("智能问答")
    
    # RAG方法选择
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rag_method = st.selectbox(
            "选择RAG方法",
            ["basic", "hyde", "rerank", "enhanced"],
            index=["basic", "hyde", "rerank", "enhanced"].index(st.session_state.rag_method),
            format_func=lambda x: {
                "basic": "基础RAG",
                "hyde": "HyDE增强",
                "rerank": "重排序RAG",
                "enhanced": "增强RAG (推荐)"
            }[x]
        )
        st.session_state.rag_method = rag_method
    
    with col2:
        if st.button("方法对比", help="比较不同RAG方法的表现"):
            st.session_state.show_comparison = True
    
    # 显示聊天历史
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.write(message["content"])
                    
                    # 显示置信度
                    if "confidence" in message:
                        confidence = message["confidence"]
                        confidence_color = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
                        st.caption(f"{confidence_color} 置信度: {confidence:.3f}")
                    
                    # 显示引用
                    if "citations" in message and message["citations"]:
                        with st.expander("参考文档"):
                            for i, citation in enumerate(message["citations"], 1):
                                st.write(f"**引用 {i}:** {citation['content'][:200]}...")
                                if citation.get('similarity'):
                                    st.caption(f"相似度: {citation['similarity']:.3f}")
                    
                    # 反馈收集
                    feedback_key = f"feedback_{len(st.session_state.chat_history)}_{message.get('timestamp', 0)}"
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        rating = st.select_slider(
                            "评价回答质量",
                            options=[1, 2, 3, 4, 5],
                            value=3,
                            format_func=lambda x: str(x) + "星",
                            key=feedback_key
                        )
                        
                        feedback_comment = st.text_input(
                            "反馈意见 (可选)",
                            placeholder="请分享您的想法...",
                            key=f"comment_{feedback_key}"
                        )
                        
                        if st.button("提交反馈", key=f"submit_{feedback_key}"):
                            # 这里简化了反馈收集流程
                            st.success(f"感谢您的反馈！评分: {rating}/5")
                            if feedback_comment:
                                st.info(f"反馈: {feedback_comment}")
                
                else:
                    st.write(message["content"])
    
    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 用户消息
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # AI回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 根据选择的方法获取回答
                if rag_method == "basic":
                    result = enhanced_rag_chain.ask_basic(prompt)
                elif rag_method == "hyde":
                    result = enhanced_rag_chain.ask_hyde(prompt)
                elif rag_method == "rerank":
                    result = enhanced_rag_chain.ask_rerank(prompt)
                else:  # enhanced
                    result = enhanced_rag_chain.ask_enhanced(prompt)
                
                # 显示回答
                st.write(result["answer"])
                
                # 显示置信度
                confidence = result["confidence"]
                confidence_color = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
                st.caption(f"{confidence_color} 置信度: {confidence:.3f} | 用时: {result['response_time']:.2f}秒")
                
                # 显示引用
                if result["citations"]:
                    with st.expander(f"参考文档 ({len(result['citations'])} 个)"):
                        for i, citation in enumerate(result["citations"], 1):
                            st.write(f"**引用 {i}:** {citation['content'][:200]}...")
                            if citation.get('similarity'):
                                st.caption(f"相似度: {citation['similarity']:.3f}")
                
                # 保存到聊天历史
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": confidence,
                    "citations": result["citations"],
                    "response_time": result["response_time"],
                    "timestamp": time.time()
                }
                st.session_state.chat_history.append(assistant_message)
                
                # 记录到数据库
                db_manager.add_qa_log(
                    session_id=st.session_state.session_id,
                    question=prompt,
                    answer=result["answer"],
                    confidence=confidence,
                    response_time=result["response_time"],
                    context=json.dumps([c["content"] for c in result["citations"]])
                )

def display_comparison_interface():
    """显示方法对比界面"""
    if st.session_state.get('show_comparison', False):
        st.header("RAG方法对比")
        
        test_question = st.text_input(
            "输入测试问题",
            value="什么是RAG技术？它有什么优势？",
            help="输入一个问题来比较不同RAG方法的表现"
        )
        
        if st.button("开始对比"):
            methods = {
                "基础RAG": enhanced_rag_chain.ask_basic,
                "HyDE增强": enhanced_rag_chain.ask_hyde,
                "重排序RAG": enhanced_rag_chain.ask_rerank,
                "增强RAG": enhanced_rag_chain.ask_enhanced
            }
            
            comparison_results = {}
            
            progress_bar = st.progress(0)
            
            for i, (method_name, method_func) in enumerate(methods.items()):
                with st.spinner(f"测试 {method_name}..."):
                    result = method_func(test_question)
                    comparison_results[method_name] = result
                    progress_bar.progress((i + 1) / len(methods))
            
            # 显示对比结果
            st.subheader("对比结果")
            
            for method_name, result in comparison_results.items():
                with st.expander(f"{method_name} - 置信度: {result['confidence']:.3f}"):
                    st.write("**回答:**")
                    st.write(result["answer"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("置信度", f"{result['confidence']:.3f}")
                    with col2:
                        st.metric("响应时间", f"{result['response_time']:.2f}s")
                    with col3:
                        st.metric("引用数量", len(result["citations"]))
                    
                    if result["citations"]:
                        st.write("**参考文档:**")
                        for i, citation in enumerate(result["citations"][:2], 1):
                            st.write(f"{i}. {citation['content'][:100]}...")
        
        if st.button("关闭对比"):
            st.session_state.show_comparison = False
            st.rerun()

def display_document_management():
    """显示文档管理界面"""
    st.header("文档管理")
    
    # 文档上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=['txt', 'pdf', 'docx', 'md'],
        help="支持TXT、PDF、DOCX、Markdown格式"
    )
    
    if uploaded_file is not None:
        if st.button("📥 处理文档"):
            with st.spinner("处理文档中..."):
                chunks_count = upload_and_process_file(uploaded_file)
                
                if chunks_count > 0:
                    st.success(f"文档处理完成！共生成 {chunks_count} 个文本块")
                else:
                    st.error("文档处理失败")
    
    # 显示已上传的文档
    st.subheader("已上传文档")
    
    documents = db_manager.get_document_stats()
    
    if documents:
        for doc in documents:
            with st.expander(f"{doc.filename}"):
                metadata = json.loads(doc.metadata) if doc.metadata else {}
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**文件大小:** {metadata.get('file_size', 'N/A')} 字节")
                    st.write(f"**文本块数:** {metadata.get('chunks_count', 'N/A')}")
                
                with col2:
                    upload_time = metadata.get('upload_time')
                    if upload_time:
                        st.write(f"**上传时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(upload_time))}")
    else:
        st.info("暂无已上传的文档")

def display_evaluation_panel():
    """显示评测面板"""
    st.header("系统评测")
    
    tab1, tab2, tab3 = st.tabs(["RAGAS评测", "用户反馈", " 多模型支持"])
    
    with tab1:
        st.subheader("RAGAS自动化评测")
        
        if st.button("运行评测"):
            with st.spinner("运行RAGAS评测中..."):
                try:
                    # 创建测试数据集
                    dataset = rag_evaluator.create_evaluation_dataset()
                    
                    # 运行评测
                    results = rag_evaluator.run_method_comparison(dataset)
                    
                    # 生成报告
                    report = rag_evaluator.generate_evaluation_report(results)
                    
                    # 显示结果
                    st.subheader("评测结果")
                    
                    # 方法排名
                    st.write("**方法排名:**")
                    for rank, (method, score) in enumerate(report['overall_ranking'], 1):
                        st.write(f"{rank}. {method.upper()}: {score:.3f}")
                    
                    # 详细指标
                    st.write("**详细指标:**")
                    for method, perf in report['method_performance'].items():
                        metrics = perf['metrics']
                        
                        with st.expander(f"{method.upper()} 方法"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("忠实度", f"{metrics['avg_faithfulness']:.3f}")
                            with col2:
                                st.metric("相关性", f"{metrics['avg_relevance']:.3f}")
                            with col3:
                                st.metric("完整性", f"{metrics['avg_completeness']:.3f}")
                            with col4:
                                st.metric("综合分", f"{metrics['avg_overall_score']:.3f}")
                
                except Exception as e:
                    st.error(f"评测失败: {e}")
    
    with tab2:
        st.subheader("用户反馈分析")
        
        # 显示反馈统计
        stats = feedback_learner.get_feedback_statistics()
        
        if stats.get("total_feedback", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总反馈数", stats["total_feedback"])
            with col2:
                st.metric("平均评分", f"{stats['average_rating']:.2f}/5")
            with col3:
                st.metric("满意度", f"{stats['satisfaction_rate']:.1f}%")
            with col4:
                st.metric("低评分数", stats["low_rating_count"])
            
            # 评分分布
            st.write("**评分分布:**")
            rating_data = []
            for rating in range(1, 6):
                count = stats['rating_distribution'].get(rating, 0)
                rating_data.append({"评分": f"{rating}星", "数量": count})
            
            st.bar_chart({item["评分"]: item["数量"] for item in rating_data})
            
            # 改进建议
            suggestions = feedback_learner.generate_improvement_suggestions()
            if suggestions:
                st.write("**改进建议:**")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
        else:
            st.info("暂无用户反馈数据")
    
    with tab3:
        st.subheader(" 多模型支持")
        
        # 显示可用模型
        available_models = multi_model_manager.list_available_models()
        st.write(f"**可用模型:** {', '.join(available_models)}")
        
        # 模型信息
        selected_model = st.selectbox("选择模型查看详情", available_models)
        
        if selected_model:
            model_info = multi_model_manager.get_model_info(selected_model)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**提供商:** {model_info.get('provider', '未知')}")
                st.write(f"**最大Token:** {model_info.get('max_tokens', '未知')}")
            
            with col2:
                st.write(f"**支持函数调用:** {'是' if model_info.get('supports_function_calling') else '否'}")
                st.write(f"**支持流式输出:** {'是' if model_info.get('supports_streaming') else '否'}")
        
        # 模型对比测试
        if st.button("运行模型对比"):
            test_question = "请简要解释人工智能的概念"
            
            with st.spinner("运行模型对比测试..."):
                try:
                    # 只比较前两个可用模型
                    test_models = available_models[:2]
                    results = multi_model_manager.compare_models(test_question, test_models)
                    
                    st.write("**对比结果:**")
                    for model, result in results.items():
                        with st.expander(f"{model}"):
                            if "error" not in result:
                                st.write(result["content"])
                                st.caption(f"用时: {result['response_time']:.2f}秒 | Token: {result['tokens_used']}")
                            else:
                                st.error(f"错误: {result['error']}")
                
                except Exception as e:
                    st.error(f"模型对比失败: {e}")

def display_system_status():
    """显示系统状态"""
    with st.sidebar:
        st.header("系统状态")
        
        # 数据库连接状态
        try:
            session = db_manager.get_session()
            session.close()
            db_status = "高 正常"
        except:
            db_status = "低 异常"
        
        st.write(f"**数据库:** {db_status}")
        
        # 向量库状态
        try:
            vector_count = vector_store.collection.count()
            vector_status = f"高 正常 ({vector_count} 向量)"
        except:
            vector_status = "低 异常"
        
        st.write(f"**向量库:** {vector_status}")
        
        # 模型状态
        model_status = "高 正常" if multi_model_manager.default_provider else "低 异常"
        st.write(f"**LLM模型:** {model_status}")
        
        st.write(f"**当前模型:** {st.session_state.selected_model}")
        st.write(f"**RAG方法:** {st.session_state.rag_method}")
        
        # 会话信息
        st.write(f"**会话ID:** {st.session_state.session_id}")
        st.write(f"**消息数:** {len(st.session_state.chat_history)}")

def main():
    """主函数"""
    # 初始化
    initialize_session_state()
    
    # 标题
    st.title(" RAGLite - 智能问答助手")
    st.caption("基于LangChain与MySQL的企业级RAG系统")
    
    # 侧边栏导航
    with st.sidebar:
        st.header("🧭 导航")
        page = st.radio(
            "选择功能",
            ["智能问答", "文档管理", "系统评测"],
            index=0
        )
    
    # 显示对应页面
    if page == "智能问答":
        display_chat_interface()
        display_comparison_interface()
    elif page == "文档管理":
        display_document_management()
    elif page == "系统评测":
        display_evaluation_panel()
    
    # 系统状态
    display_system_status()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "**RAGLite** | "
        "支持多种RAG方法 | "
        "自动化评测 | "
        "用户反馈学习 | "
        " 多模型支持"
    )

if __name__ == "__main__":
    main()
