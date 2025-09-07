"""
RAGLite 一键启动脚本
自动运行完整的RAGLite系统
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Python版本过低，需要Python 3.8+")
        return False
    
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        "streamlit",
        "openai",
        "sqlalchemy",
        "chromadb",
        "langchain",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "python-dotenv":
                __import__("dotenv")
            else:
                __import__(package.replace("-", "_"))
            print(f"{package}")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} 未安装")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_configuration():
    """检查配置文件"""
    print("\n检查配置文件...")
    
    config_file = Path("config/settings.py")
    if not config_file.exists():
        print("配置文件不存在")
        return False
    
    # 检查环境变量文件
    env_file = Path(".env")
    if not env_file.exists():
        print(".env文件不存在，将使用默认配置")
        create_default_env()
    
    try:
        from config.settings import settings
        
        # 检查关键配置
        if not settings.OPENAI_API_KEY:
            print("未配置OpenAI API Key")
            return False
        
        print("配置文件检查通过")
        return True
        
    except Exception as e:
        print(f"配置文件错误: {e}")
        return False

def create_default_env():
    """创建默认环境变量文件"""
    default_env = """# OpenAI配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# MySQL数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=raglite_db

# RAG配置
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
TOP_K=5
TEMPERATURE=0.7
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(default_env)
    
    print("已创建默认.env文件，请修改其中的配置")

def initialize_database():
    """初始化数据库"""
    print("\n初始化数据库...")
    
    try:
        from database.db_manager import db_manager
        
        # 创建表
        result = db_manager.create_tables()
        if result:
            print("数据库表创建成功")
        else:
            print("数据库表可能已存在")
        
        # 测试连接
        session = db_manager.get_session()
        session.close()
        print("数据库连接测试成功")
        
        return True
        
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False

def initialize_vector_store():
    """初始化向量存储"""
    print("\n初始化向量存储...")
    
    try:
        from core.vector_store_compatible import vector_store
        from langchain.schema import Document
        
        # 测试向量存储
        test_docs = [Document(page_content="这是一个测试文档", metadata={"source": "test"})]
        vector_store.add_documents(test_docs)
        
        # 获取统计信息
        stats = vector_store.get_stats()
        count = stats.get('total_documents', 0)
        print(f"向量存储初始化成功 (当前向量数: {count})")
        
        return True
        
    except Exception as e:
        print(f"向量存储初始化失败: {e}")
        return False

def run_system_tests():
    """运行系统测试"""
    print("\n运行系统测试...")
    
    try:
        # 测试基础RAG功能
        from core.enhanced_rag_chain import enhanced_rag_chain
        
        test_question = "什么是人工智能？"
        result = enhanced_rag_chain.ask_basic(test_question)
        
        if result and result.get("answer"):
            print("基础RAG功能测试通过")
        else:
            print("基础RAG功能测试失败")
            return False
        
        # 测试增强RAG功能
        result = enhanced_rag_chain.ask_enhanced(test_question)
        
        if result and result.get("answer"):
            print("增强RAG功能测试通过")
        else:
            print("增强RAG功能测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"系统测试失败: {e}")
        return False

def start_streamlit_app():
    """启动Streamlit应用"""
    print("\n启动RAGLite Web应用...")
    
    try:
        # 选择应用版本
        print("启动 Web应用 (app.py)...")
        print("Web应用将在浏览器中打开...")
        print("使用Ctrl+C停止应用")
        
        app_file = "app.py"
        
        # 启动Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n应用已停止")
        return True
    except Exception as e:
        print(f"启动应用失败: {e}")
        return False

def run_tests():
    """运行测试套件"""
    print("\n运行测试套件...")
    
    test_scripts = [
        ("基础功能", "test_basic.py"),
        ("增强功能", "test_enhanced.py"),
        ("高级功能", "test_advanced.py")
    ]
    
    for test_name, test_script in test_scripts:
        if os.path.exists(test_script):
            print(f"\n运行 {test_name} 测试...")
            
            choice = input(f"是否运行 {test_name} 测试？(y/n): ").lower()
            if choice == 'y':
                try:
                    subprocess.run([sys.executable, test_script], check=True)
                    print(f"{test_name} 测试完成")
                except subprocess.CalledProcessError:
                    print(f"{test_name} 测试失败")
                except KeyboardInterrupt:
                    print(f"{test_name} 测试被中断")
        else:
            print(f"{test_script} 不存在")

def display_system_info():
    """显示系统信息"""
    print("\n" + "="*60)
    print("RAGLite 系统信息")
    print("="*60)
    
    try:
        # 数据库状态
        from database.db_manager import db_manager
        stats = db_manager.get_conversation_stats()
        print(f"对话统计: {stats}")
        
        # 向量库状态
        from core.vector_store_compatible import vector_store
        vector_count = vector_store.collection.count()
        print(f"向量数量: {vector_count}")
        
        # 可用模型
        from evaluation.multi_model_support import multi_model_manager
        models = multi_model_manager.list_available_models()
        print(f"可用模型: {', '.join(models)}")
        
        print("\n功能模块:")
        print("   基础RAG")
        print("   HyDE检索增强")
        print("   文档重排序")
        print("   增强RAG链")
        print("   RAGAS自动评测")
        print("   用户反馈学习")
        print("   多模型支持")
        print("   Web界面")
        
    except Exception as e:
        print(f"获取系统信息失败: {e}")

def main():
    """主函数"""
    print("RAGLite - 智能问答助手启动器")
    print("=" * 50)
    
    # 切换到脚本目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 启动菜单
    while True:
        print("\nRAGLite 启动菜单")
        print("-" * 30)
        print("1. 环境检查")
        print("2. 初始化系统")
        print("3. 运行测试")
        print("4. 启动Web应用")
        print("5. 显示系统信息")
        print("6. 一键启动 (推荐)")
        print("0. 退出")
        
        choice = input("\n请选择操作 (0-6): ").strip()
        
        if choice == "0":
            print("再见！")
            break
        
        elif choice == "1":
            check_environment()
            check_configuration()
        
        elif choice == "2":
            if check_environment() and check_configuration():
                initialize_database()
                initialize_vector_store()
        
        elif choice == "3":
            run_tests()
        
        elif choice == "4":
            start_streamlit_app()
        
        elif choice == "5":
            display_system_info()
        
        elif choice == "6":
            print("一键启动RAGLite...")
            
            # 完整的启动流程
            if (check_environment() and 
                check_configuration() and 
                initialize_database() and 
                initialize_vector_store() and 
                run_system_tests()):
                
                print("\n系统初始化完成！")
                display_system_info()
                
                print("\n启动Web应用...")
                start_streamlit_app()
            else:
                print("\n系统初始化失败，请检查错误信息")
        
        else:
            print("无效选择，请重新输入")
        
        if choice != "4":  # 不是启动应用的选项
            input("\n按回车键继续...")

if __name__ == "__main__":
    main()
