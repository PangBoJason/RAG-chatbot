from config.settings import settings

def test_settings():
    """测试配置"""
    print("测试配置...")
    print(f"API Key前10位: {settings.OPENAI_API_KEY[:10] if settings.OPENAI_API_KEY else 'None'}***")
    print(f"向量存储目录: {settings.CHROMA_PERSIST_DIR}")

def test_openai_direct():
    """直接测试OpenAI API"""
    try:
        import openai
        
        client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
        # 测试向量化
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="测试文本"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI API测试成功！向量维度: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ OpenAI API测试失败: {e}")

def test_compatible_embeddings():
    """测试兼容的向量化类"""
    try:
        from core.vector_store_compatible import CompatibleEmbeddings
        
        embeddings = CompatibleEmbeddings()
        result = embeddings.embed_query("测试文本")
        print(f"✅ 兼容向量化成功！向量维度: {len(result)}")
        
    except Exception as e:
        print(f"❌ 兼容向量化失败: {e}")

if __name__ == "__main__":
    test_settings()
    print()
    test_openai_direct()
    print()
    test_compatible_embeddings()