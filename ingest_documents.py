from core.document_loader import document_loader
from core.vector_store_compatible import vector_store
from database.db_manager import db_manager
import os
import time
import argparse

def ingest_document(file_path: str):
    """入库单个文档"""
    print(f"\n=== 开始处理文档: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        start_time = time.time()
        
        # 1. 处理文档
        chunks = document_loader.process_document(file_path)
        
        # 2. 向量化存储
        vector_ids = vector_store.add_documents(chunks)
        
        # 3. 记录到数据库
        file_size = os.path.getsize(file_path)
        processing_time = time.time() - start_time
        
        doc_id = db_manager.log_document(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size=file_size,
            chunk_count=len(chunks),
            processing_time=processing_time
        )
        
        print(f"✅ 文档入库成功!")
        print(f"   - 文档ID: {doc_id}")
        print(f"   - 分块数: {len(chunks)}")
        print(f"   - 向量数: {len(vector_ids)}")
        print(f"   - 处理时间: {processing_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 文档入库失败: {e}")
        return False

def ingest_directory(directory: str):
    """入库目录下的所有文档"""
    print(f"\n=== 开始批量入库: {directory} ===")
    
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    # 支持的文件格式
    supported_formats = ['.txt', '.pdf', '.md']
    
    # 找到所有支持的文件
    files_to_process = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"❌ 在 {directory} 中没有找到支持的文档文件")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return
    
    print(f"📁 找到 {len(files_to_process)} 个文档文件:")
    for file_path in files_to_process:
        print(f"   - {file_path}")
    
    # 批量处理
    success_count = 0
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n进度: {i}/{len(files_to_process)}")
        
        if ingest_document(file_path):
            success_count += 1
    
    print(f"\n🎉 批量入库完成!")
    print(f"📊 成功处理: {success_count}/{len(files_to_process)} 个文档")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAGLite 文档入库工具")
    parser.add_argument("path", help="文档文件路径或目录路径")
    parser.add_argument("--batch", action="store_true", help="批量处理目录下的所有文档")
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.path):
        ingest_directory(args.path)
    else:
        ingest_document(args.path)

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式选择
    import sys
    
    if len(sys.argv) == 1:
        print("=== RAGLite 文档入库工具 ===")
        print("1. 入库单个文档")
        print("2. 批量入库目录")
        
        choice = input("请选择模式 (1/2): ").strip()
        
        if choice == "1":
            file_path = input("请输入文档路径: ").strip()
            ingest_document(file_path)
        elif choice == "2":
            directory = input("请输入目录路径: ").strip()
            ingest_directory(directory)
        else:
            print("无效选择")
    else:
        main()
