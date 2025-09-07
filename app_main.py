"""
RAGLite：基于LangChain与MySQL的智能问答助手

主要特性：
1. RAG 检索问答（LangChain + Chroma）
2. MySQL 存储问答 & 文档元数据
3. HyDE 检索增强 + Rerank 优化
4. 可视化引用片段，答案可解释
5. 用户反馈闭环系统
6. 多模型切换支持
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import tempfile
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# 导入核心模块
from core.enhanced_rag_chain import enhanced_rag_chain
from core.hyde_retrieval import hyde_retriever
from core.reranker import reranker
from core.document_loader import document_loader
from core.vector_store_compatible import vector_store
from database.db_manager import db_manager
from evaluation.rag_evaluator_new import rag_evaluator
from config.settings import settings

# 页面配置
st.set_page_config(
    page_title="RAGLite - 智能问答助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* 统计卡片 */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* 反馈按钮 */
    .feedback-container {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .feedback-btn {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .feedback-like {
        background: #d4edda;
        color: #155724;
    }
    
    .feedback-dislike {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* 引用片段样式 */
    .citation {
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
    }
    
    .citation-source {
        font-weight: bold;
        color: #667eea;
        font-size: 0.9rem;
    }
    
    .citation-content {
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* 隐藏空白元素 */
    div:empty { display: none !important; }
    .element-container:empty { display: none !important; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """初始化会话状态"""
    if 'session_id' not in st.session_state:
        try:
            session_id = db_manager.create_conversation("用户")
            st.session_state.session_id = session_id
            print(f"✅ 创建新会话: {session_id}")
        except Exception as e:
            print(f"❌ 创建会话失败: {e}")
            import time
            backup_session_id = f"backup_{int(time.time())}"
            st.session_state.session_id = backup_session_id
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'retrieval_mode' not in st.session_state:
        st.session_state.retrieval_mode = "enhanced"
    if 'current_model' not in st.session_state:
        st.session_state.current_model = settings.MODEL_NAME
    if 'show_citations' not in st.session_state:
        st.session_state.show_citations = True
    if 'enable_rerank' not in st.session_state:
        st.session_state.enable_rerank = True

def render_header():
    """渲染项目头部"""
    st.markdown(f"""
    <div class="main-header">
        <div class="main-title">🤖 RAGLite</div>
        <div class="main-subtitle">基于LangChain与MySQL的智能问答助手</div>
        <div class="feature-badges">
            <span class="badge">🔍 RAG检索问答</span>
            <span class="badge">🗄️ MySQL存储</span>
            <span class="badge">💡 HyDE增强</span>
            <span class="badge">🎯 Rerank优化</span>
            <span class="badge">📊 可解释AI</span>
            <span class="badge">👥 用户反馈</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_system_stats():
    """获取系统统计数据"""
    try:
        # 获取问答统计
        qa_stats = db_manager.get_qa_stats()
        
        # 获取文档统计
        doc_stats = db_manager.get_document_stats()
        
        # 获取向量库统计
        vector_stats = vector_store.get_stats()
        
        # 获取反馈统计
        feedback_stats = db_manager.get_feedback_stats()
        
        return {
            "total_questions": qa_stats.get("total_count", 0),
            "total_documents": doc_stats.get("total_count", 0),
            "vector_chunks": vector_stats.get("total_documents", 0),
            "avg_confidence": qa_stats.get("avg_confidence", 0.0),
            "positive_feedback": feedback_stats.get("positive_count", 0),
            "total_feedback": feedback_stats.get("total_count", 0)
        }
    except Exception as e:
        print(f"获取统计数据失败: {e}")
        return {
            "total_questions": 0,
            "total_documents": 0,
            "vector_chunks": 0,
            "avg_confidence": 0.0,
            "positive_feedback": 0,
            "total_feedback": 0
        }

def render_dashboard():
    """渲染仪表板"""
    stats = get_system_stats()
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 总问答数",
            value=stats["total_questions"],
            help="系统处理的总问答数量"
        )
    
    with col2:
        st.metric(
            label="📚 知识文档",
            value=f"{stats['total_documents']} / {stats['vector_chunks']}",
            help="文档数量 / 向量块数量"
        )
    
    with col3:
        confidence_pct = stats["avg_confidence"] * 100 if stats["avg_confidence"] else 0
        st.metric(
            label="🎯 平均置信度",
            value=f"{confidence_pct:.1f}%",
            help="AI回答的平均置信度"
        )
    
    with col4:
        if stats["total_feedback"] > 0:
            satisfaction = stats["positive_feedback"] / stats["total_feedback"] * 100
            st.metric(
                label="👍 用户满意度",
                value=f"{satisfaction:.1f}%",
                help="用户正向反馈比例"
            )
        else:
            st.metric(
                label="👍 用户满意度",
                value="暂无数据",
                help="用户正向反馈比例"
            )

def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 检索策略
        st.subheader("🔍 检索策略")
        retrieval_mode = st.selectbox(
            "选择检索模式",
            options=["basic", "hyde", "rerank", "enhanced"],
            index=["basic", "hyde", "rerank", "enhanced"].index(st.session_state.retrieval_mode),
            format_func=lambda x: {
                "basic": "🔍 基础检索",
                "hyde": "💡 HyDE增强",
                "rerank": "🎯 重排序",
                "enhanced": "🚀 智能增强"
            }[x]
        )
        st.session_state.retrieval_mode = retrieval_mode
        
        # 策略描述
        descriptions = {
            "basic": "传统向量相似度检索",
            "hyde": "生成假设答案增强检索召回",
            "rerank": "多阶段检索与智能重排序",
            "enhanced": "综合HyDE + Rerank的最佳策略"
        }
        st.caption(descriptions[retrieval_mode])
        
        # Rerank开关
        enable_rerank = st.checkbox(
            "启用Rerank重排序",
            value=st.session_state.enable_rerank,
            help="使用BGE模型对检索结果重新排序"
        )
        st.session_state.enable_rerank = enable_rerank
        
        st.divider()
        
        # 模型配置
        st.subheader("🤖 模型配置")
        
        # 可用模型列表
        available_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "deepseek-chat",
            "qwen-max"
        ]
        
        current_model = st.selectbox(
            "选择LLM模型",
            options=available_models,
            index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
        )
        st.session_state.current_model = current_model
        
        # 显示设置
        st.divider()
        st.subheader("📱 显示设置")
        
        show_citations = st.checkbox(
            "显示引用片段",
            value=st.session_state.show_citations,
            help="在回答中显示引用的文档片段"
        )
        st.session_state.show_citations = show_citations
        
        # 文件上传
        st.divider()
        st.subheader("📁 文档管理")
        
        uploaded_files = st.file_uploader(
            "上传知识文档",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="支持多格式文档上传"
        )
        
        if uploaded_files:
            if st.button("🚀 处理文档", use_container_width=True):
                process_documents(uploaded_files)
        
        # 系统管理
        st.divider()
        st.subheader("🛠️ 系统管理")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📊 查看统计"):
                st.session_state.show_stats = True
        
        # 评测功能
        if st.button("🧪 运行评测", use_container_width=True):
            run_evaluation()

def process_documents(uploaded_files):
    """处理上传的文档"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, file in enumerate(uploaded_files):
            progress = i / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"处理文档: {file.name}")
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # 记录文档元数据
                doc_metadata = db_manager.create_document_metadata(
                    filename=file.name,
                    file_size=len(file.getvalue())
                )
                doc_id = doc_metadata.id  # 立即获取ID
                
                # 处理文档
                start_time = time.time()
                chunks = document_loader.process_document(tmp_path)
                processing_time = time.time() - start_time
                
                if chunks:
                    # 添加到向量库
                    vector_store.add_documents(chunks)
                    
                    # 更新元数据
                    db_manager.update_document_metadata(
                        doc_id,
                        chunk_count=len(chunks),
                        processing_time=processing_time,
                        is_processed=True
                    )
                    
                    st.success(f"✅ {file.name} 处理成功 ({len(chunks)}块)")
                
            finally:
                os.unlink(tmp_path)
        
        progress_bar.progress(1.0)
        status_text.text("处理完成！")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"文档处理失败: {e}")

def run_evaluation():
    """运行RAG评测"""
    with st.spinner("正在运行RAGAS评测..."):
        try:
            # 获取最近的问答记录
            recent_qa = db_manager.get_recent_qa_for_evaluation(limit=10)
            
            if not recent_qa:
                st.warning("没有足够的问答数据进行评测")
                return
            
            # 运行评测
            eval_results = rag_evaluator.evaluate_qa_batch(recent_qa)
            
            # 保存评测结果
            eval_id = db_manager.save_evaluation_results(eval_results)
            
            st.success(f"✅ 评测完成！评测ID: {eval_id}")
            
            # 显示结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("忠实度", f"{eval_results['faithfulness']:.3f}")
            with col2:
                st.metric("相关性", f"{eval_results['answer_relevancy']:.3f}")
            with col3:
                st.metric("召回率", f"{eval_results['context_recall']:.3f}")
            
        except Exception as e:
            st.error(f"评测失败: {e}")

def render_chat_interface():
    """渲染聊天界面"""
    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # 显示AI回答
                st.write(message["content"])
                
                # 显示置信度
                if "confidence" in message:
                    confidence = message["confidence"]
                    color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.markdown(f"<span style='color: {color}'>🎯 置信度: {confidence:.2f}</span>", unsafe_allow_html=True)
                
                # 显示引用片段
                if st.session_state.show_citations and message.get("citations"):
                    with st.expander(f"📚 查看引用片段 ({len(message['citations'])}个)"):
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(f"""
                            <div class="citation">
                                <div class="citation-source">片段 {i}: {citation.get('source', '未知来源')}</div>
                                <div class="citation-content">{citation['content'][:300]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # 用户反馈按钮
                col1, col2, col3 = st.columns([1, 1, 6])
                
                with col1:
                    if st.button("👍", key=f"like_{message['id']}"):
                        add_feedback(message["id"], "like")
                        st.success("感谢您的反馈！")
                
                with col2:
                    if st.button("👎", key=f"dislike_{message['id']}"):
                        add_feedback(message["id"], "dislike")
                        # 可以添加困难问题收集逻辑
                        st.warning("已记录问题，我们会改进")
            
            else:
                st.write(message["content"])
    
    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.chat_history.append({"role": "user", "content": prompt, "id": f"user_{len(st.session_state.chat_history)}"})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # AI处理
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                try:
                    # 根据策略选择RAG方法
                    if st.session_state.retrieval_mode == "basic":
                        result = enhanced_rag_chain.ask_basic(prompt)
                    elif st.session_state.retrieval_mode == "hyde":
                        result = enhanced_rag_chain.ask_hyde(prompt)
                    elif st.session_state.retrieval_mode == "rerank":
                        result = enhanced_rag_chain.ask_rerank(prompt)
                    else:  # enhanced
                        result = enhanced_rag_chain.ask_enhanced(prompt)
                    
                    # 显示回答
                    st.write(result["answer"])
                    
                    # 显示置信度
                    confidence = result.get("confidence", 0.0)
                    color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.markdown(f"<span style='color: {color}'>🎯 置信度: {confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    # 显示引用
                    if st.session_state.show_citations and result.get("citations"):
                        with st.expander(f"📚 查看引用片段 ({len(result['citations'])}个)"):
                            for i, citation in enumerate(result["citations"], 1):
                                st.markdown(f"""
                                <div class="citation">
                                    <div class="citation-source">片段 {i}: {citation.get('source', '未知来源')}</div>
                                    <div class="citation-content">{citation['content'][:300]}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # 保存到历史
                    message_id = f"assistant_{len(st.session_state.chat_history)}"
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "confidence": confidence,
                        "citations": result.get("citations", []),
                        "response_time": result.get("response_time", 0.0),
                        "id": message_id
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # 记录到数据库
                    try:
                        qa_id = db_manager.log_qa(
                            session_id=st.session_state.session_id,
                            question=prompt,
                            answer=result["answer"],
                            citations=result.get("citations", []),
                            confidence=confidence,
                            response_time=result.get("response_time", 0.0),
                            model_name=st.session_state.current_model
                        )
                        print(f"✅ 记录到数据库: QA ID {qa_id}")
                        
                        # 更新message_id为数据库ID
                        st.session_state.chat_history[-1]["id"] = qa_id
                        
                    except Exception as e:
                        print(f"❌ 数据库记录失败: {e}")
                        st.error(f"数据库记录失败: {e}")
                
                except Exception as e:
                    st.error(f"处理请求失败: {e}")

def add_feedback(qa_id: int, feedback_type: str):
    """添加用户反馈"""
    try:
        db_manager.add_feedback(qa_id, feedback_type)
        
        # 如果是负面反馈，添加到困难样本
        if feedback_type == "dislike":
            # 这里可以实现困难样本收集逻辑
            pass
            
    except Exception as e:
        print(f"反馈记录失败: {e}")

def render_analytics_page():
    """渲染分析页面"""
    st.header("📊 系统分析")
    
    # 获取分析数据
    analytics_data = db_manager.get_analytics_data()
    
    # 问答趋势
    st.subheader("📈 问答趋势")
    if analytics_data.get("daily_qa"):
        df_qa = pd.DataFrame(analytics_data["daily_qa"])
        st.line_chart(df_qa.set_index("date")["count"])
    
    # 置信度分布
    st.subheader("🎯 置信度分布")
    if analytics_data.get("confidence_distribution"):
        df_conf = pd.DataFrame(analytics_data["confidence_distribution"])
        st.bar_chart(df_conf.set_index("range")["count"])
    
    # 用户反馈统计
    st.subheader("👥 用户反馈")
    if analytics_data.get("feedback_stats"):
        feedback_stats = analytics_data["feedback_stats"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("正向反馈", feedback_stats.get("positive", 0))
        with col2:
            st.metric("负向反馈", feedback_stats.get("negative", 0))

def main():
    """主函数"""
    initialize_session_state()
    
    # 渲染页面
    render_header()
    render_dashboard()
    
    # 侧边栏
    render_sidebar()
    
    # 主界面选项卡
    tab1, tab2, tab3 = st.tabs(["💬 智能问答", "📊 数据分析", "🧪 系统评测"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_analytics_page()
    
    with tab3:
        st.header("🧪 RAGAS评测系统")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("运行完整评测", use_container_width=True):
                run_evaluation()
        
        with col2:
            if st.button("查看历史评测", use_container_width=True):
                # 显示历史评测结果
                eval_history = db_manager.get_evaluation_history()
                if eval_history:
                    st.dataframe(eval_history)
                else:
                    st.info("暂无评测历史")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🤖 <strong>RAGLite</strong> - 基于LangChain与MySQL的智能问答助手<br>
            <small>集成 RAG检索 | HyDE增强 | Rerank优化 | 用户反馈 | 自动评测</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
