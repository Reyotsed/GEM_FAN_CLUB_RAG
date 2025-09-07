if __name__ == "__main__":
    import rag_modules.data_preparation
    import config
    
    # 加载数据准备模块，执行数据加载、清洗、分块和向量数据库创建流程
    dp = rag_modules.data_preparation.DataPreparationModule(config.DATA_PATH)

    dp.load_data()

    dp.chunk_documents()

    dp.create_vector_db()

