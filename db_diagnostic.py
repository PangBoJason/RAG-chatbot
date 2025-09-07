"""
数据库诊断和初始化工具
检查MySQL连接、创建表、测试数据插入
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from config.settings import settings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql

def test_mysql_connection():
    """测试MySQL基础连接"""
    print("=== 测试MySQL连接 ===")
    
    try:
        # 使用pymysql直接连接
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✅ MySQL连接成功! 版本: {version[0]}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ MySQL连接失败: {e}")
        return False

def test_database_exists():
    """检查数据库是否存在"""
    print("\n=== 检查数据库 ===")
    
    try:
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            databases = [row[0] for row in cursor.fetchall()]
            
            if settings.MYSQL_DATABASE in databases:
                print(f"✅ 数据库 '{settings.MYSQL_DATABASE}' 存在")
                return True
            else:
                print(f"❌ 数据库 '{settings.MYSQL_DATABASE}' 不存在")
                print("可用数据库:", databases)
                return False
        
    except Exception as e:
        print(f"❌ 检查数据库失败: {e}")
        return False

def create_database():
    """创建数据库"""
    print("\n=== 创建数据库 ===")
    
    try:
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ 数据库 '{settings.MYSQL_DATABASE}' 创建成功")
        
        connection.commit()
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ 创建数据库失败: {e}")
        return False

def test_sqlalchemy_connection():
    """测试SQLAlchemy连接"""
    print("\n=== 测试SQLAlchemy连接 ===")
    
    try:
        engine = create_engine(settings.mysql_url, echo=True)
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print(f"✅ SQLAlchemy连接成功: {result.fetchone()}")
        
        return True
        
    except Exception as e:
        print(f"❌ SQLAlchemy连接失败: {e}")
        print(f"连接URL: {settings.mysql_url}")
        return False

def create_tables():
    """创建数据表"""
    print("\n=== 创建数据表 ===")
    
    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("✅ 数据表创建成功")
        return True
        
    except Exception as e:
        print(f"❌ 创建数据表失败: {e}")
        return False

def test_data_operations():
    """测试数据操作"""
    print("\n=== 测试数据操作 ===")
    
    try:
        db_manager = DatabaseManager()
        
        # 创建会话
        session_id = db_manager.create_conversation("test_user")
        print(f"✅ 创建会话成功: {session_id}")
        
        # 记录问答
        qa_id = db_manager.log_qa(
            session_id=session_id,
            question="测试问题",
            answer="测试答案",
            confidence=0.95,
            response_time=1.5
        )
        print(f"✅ 记录问答成功: {qa_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据操作失败: {e}")
        return False

def check_existing_data():
    """检查现有数据"""
    print("\n=== 检查现有数据 ===")
    
    try:
        engine = create_engine(settings.mysql_url)
        
        with engine.connect() as connection:
            # 检查会话数
            result = connection.execute(text("SELECT COUNT(*) FROM conversations"))
            conv_count = result.fetchone()[0]
            print(f"📊 对话会话数量: {conv_count}")
            
            # 检查问答记录数
            result = connection.execute(text("SELECT COUNT(*) FROM qa_logs"))
            qa_count = result.fetchone()[0]
            print(f"📊 问答记录数量: {qa_count}")
            
            if qa_count > 0:
                # 显示最近的记录
                result = connection.execute(text("""
                    SELECT id, question, answer, created_at 
                    FROM qa_logs 
                    ORDER BY created_at DESC 
                    LIMIT 3
                """))
                
                print("\n📝 最近的问答记录:")
                for row in result:
                    print(f"  ID:{row[0]} | {row[3]} | Q:{row[1][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查数据失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 RAGLite 数据库诊断工具")
    print("=" * 50)
    
    print(f"📋 配置信息:")
    print(f"   MySQL主机: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}")
    print(f"   数据库名: {settings.MYSQL_DATABASE}")
    print(f"   用户名: {settings.MYSQL_USER}")
    print(f"   连接URL: {settings.mysql_url}")
    
    # 1. 测试基础连接
    if not test_mysql_connection():
        print("\n🛑 MySQL连接失败，请检查:")
        print("   1. MySQL服务是否运行")
        print("   2. 用户名密码是否正确")
        print("   3. 主机端口是否正确")
        return
    
    # 2. 检查数据库
    if not test_database_exists():
        print("\n🔧 尝试创建数据库...")
        if not create_database():
            return
    
    # 3. 测试SQLAlchemy连接
    if not test_sqlalchemy_connection():
        print("\n🛑 SQLAlchemy连接失败")
        return
    
    # 4. 创建表
    if not create_tables():
        return
    
    # 5. 测试数据操作
    if not test_data_operations():
        return
    
    # 6. 检查现有数据
    check_existing_data()
    
    print("\n🎉 数据库诊断完成！")
    print("💡 如果仍有问题，请检查应用中的数据库调用是否正确")

if __name__ == "__main__":
    main()
