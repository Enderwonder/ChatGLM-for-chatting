import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import torch # 確保 torch 已匯入

def format_vector_snippet(vector, n=5):
    """格式化向量片段以供顯示 (顯示前n個和後n個元素)"""
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector) # 確保是 numpy array
    if vector.ndim > 1: # 如果是多維陣列 (例如批次為1的情況)
        vector = vector.flatten()
    if vector.size == 0:
        return "[]"
    if vector.size <= 2 * n:
        return f"{vector.tolist()}"
    
    first_part = [f"{x:.4f}" for x in vector[:n]]
    last_part = [f"{x:.4f}" for x in vector[-n:]]
    return f"[{', '.join(first_part)}, ..., {', '.join(last_part)}]"

class MemoryManager:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', 
                 embedding_dim=384, 
                 index_file="chat_faiss.idx", 
                 memories_pickle_file="chat_text_memories.pkl",
                 persistent_text_file="persistent_memories.txt"):
        print("🚀 初始化記憶管理器 (詳細除錯模式)...")
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.memories_pickle_file = memories_pickle_file
        self.persistent_text_memories_file = persistent_text_file

        try:
            self.embed_model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                self.embed_model = self.embed_model.to(torch.device("cuda"))
            print(f"✅ 句子嵌入模型 '{model_name}' 載入成功。")
        except Exception as e:
            print(f"❌ 載入句子嵌入模型 '{model_name}' 失敗: {e}")
            raise

        self.text_memories = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self._load_or_rebuild_from_persistent_file()

    def _load_or_rebuild_from_persistent_file(self):
        self.index.reset() 
        self.text_memories = [] 

        if not os.path.exists(self.persistent_text_memories_file):
            print(f"ℹ️ 持久記憶檔案 '{self.persistent_text_memories_file}' 不存在。將以空記憶啟動。")
            self._save_faiss_and_pkl() 
            return

        print(f"🔄 從 '{self.persistent_text_memories_file}' 完整重建記憶...")
        loaded_memories_from_text_file = []
        try:
            with open(self.persistent_text_memories_file, 'r', encoding='utf-8') as f:
                for line in f:
                    mem_text = line.strip()
                    if mem_text:
                        loaded_memories_from_text_file.append(mem_text)
            
            unique_memories = []
            seen_memories_set = set()
            for mem in loaded_memories_from_text_file:
                if mem not in seen_memories_set:
                    unique_memories.append(mem)
                    seen_memories_set.add(mem)

            for mem_text in unique_memories:
                self._add_text_to_internal_stores(mem_text, verbose=False) # 重建時不需詳細打印每個向量
            
            self._save_faiss_and_pkl()
            print(f"✅ 從 '{self.persistent_text_memories_file}' 成功載入並重建 {len(self.text_memories)} 條記憶。")
        except Exception as e:
            print(f"❌ 從 '{self.persistent_text_memories_file}' 載入記憶時發生嚴重錯誤: {e}。將以空記憶啟動。")
            self.index.reset()
            self.text_memories = []
            self._save_faiss_and_pkl()

    def _add_text_to_internal_stores(self, text_to_remember, verbose=True):
        if not text_to_remember.strip() or text_to_remember in self.text_memories:
            return False 

        try:
            embedding = self.embed_model.encode([text_to_remember], convert_to_numpy=True, normalize_embeddings=True)
            embedding = embedding.astype('float32')
            if embedding.shape[1] != self.embedding_dim:
                print(f"❌ 嵌入維度錯誤 for '{text_to_remember[:30]}...'")
                return False
            
            if verbose: # 只有在非重建模式下 (例如 /remember 指令) 才打印詳細向量
                print(f"   向量化 \"{text_to_remember[:30]}...\" (維度: {embedding.shape}): {format_vector_snippet(embedding)}")

            self.index.add(embedding)
            self.text_memories.append(text_to_remember)
            return True
        except Exception as e:
            print(f"❌ 新增記憶 '{text_to_remember[:30]}...' 到內部 FAISS/列表時出錯: {e}")
            return False

    def add_memory(self, text_to_remember):
        if not text_to_remember.strip():
            print("⚠️ 試圖新增空白記憶，已忽略。")
            return
        
        if text_to_remember in self.text_memories:
            print(f"ℹ️ 記憶 \"{text_to_remember[:50]}...\" 已存在於當前會話，忽略。")
            return

        print(f"🧠 正在為新記憶生成嵌入: \"{text_to_remember[:50]}...\"")
        # _add_text_to_internal_stores 現在會打印向量 (如果 verbose=True, 預設)
        if self._add_text_to_internal_stores(text_to_remember, verbose=True): 
            try:
                with open(self.persistent_text_memories_file, 'a', encoding='utf-8') as f:
                    f.write(text_to_remember + "\n")
                print(f"📝 新記憶已附加到 '{self.persistent_text_memories_file}'。")
                self._save_faiss_and_pkl()
                print(f"✅ 記憶新增成功: \"{text_to_remember[:50]}...\" (目前總記憶數: {self.index.ntotal})")
            except Exception as e:
                print(f"❌ 將記憶附加到 '{self.persistent_text_memories_file}' 時出錯: {e}")
        else:
            print(f"ℹ️ 記憶 \"{text_to_remember[:50]}...\" 未能新增到內部儲存。")

    def _save_faiss_and_pkl(self):
        try:
            if self.index.ntotal == len(self.text_memories): 
                faiss.write_index(self.index, self.index_file)
                with open(self.memories_pickle_file, 'wb') as f:
                    pickle.dump(self.text_memories, f)
            else:
                print(f"⚠️ 警告: FAISS 索引數量 ({self.index.ntotal}) 與文字列表數量 ({len(self.text_memories)}) 不符，未儲存 .idx/.pkl。")
        except Exception as e:
            print(f"❌ 儲存 FAISS/PKL 相關記憶時發生錯誤: {e}")

    def save_memories_on_exit(self):
        print(f"💾 程式退出，確認記憶狀態...")
        self._save_faiss_and_pkl()
        print(f"✅ 記憶狀態已儲存。總共 {self.index.ntotal} 條記憶。")

    def search_memories(self, query_text, k=3):
        """
        搜尋與查詢文字最相關的 k 條記憶。
        返回: (retrieved_texts, query_embedding, distances)
              或在錯誤/無結果時返回 ([], None, [])
        """
        if not query_text.strip() or self.index.ntotal == 0:
            return [], None, []
        try:
            query_embedding = self.embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
            query_embedding = query_embedding.astype('float32')

            if query_embedding.shape[1] != self.embedding_dim:
                 print(f"❌ 錯誤：查詢嵌入維度 ({query_embedding.shape[1]}) 與 FAISS 期望維度 ({self.embedding_dim}) 不符。")
                 return [], None, []

            actual_k = min(k, self.index.ntotal)
            if actual_k == 0: return [], query_embedding, [] # 返回查詢向量，即使沒有結果
                
            distances, indices = self.index.search(query_embedding, actual_k)
            
            retrieved_texts = [self.text_memories[i] for i in indices[0]]
            retrieved_distances = distances[0] # 取得對應的距離陣列

            return retrieved_texts, query_embedding, retrieved_distances
        except Exception as e:
            print(f"❌ 搜尋記憶時發生錯誤: {e}")
            return [], None, []

    def get_total_memories(self):
        return self.index.ntotal

    def reload_external_memories(self):
        print(f"🔄 指令觸發：準備從 '{self.persistent_text_memories_file}' 重新載入並重建所有記憶...")
        self._load_or_rebuild_from_persistent_file()

if __name__ == '__main__':
    print("--- MemoryManager 詳細除錯模式測試 ---")
    if os.path.exists("test_faiss.idx"): os.remove("test_faiss.idx")
    if os.path.exists("test_memories.pkl"): os.remove("test_memories.pkl")
    if os.path.exists("test_persistent_memories.txt"): os.remove("test_persistent_memories.txt")

    manager = MemoryManager(index_file="test_faiss.idx", 
                            memories_pickle_file="test_memories.pkl", 
                            persistent_text_file="test_persistent_memories.txt")
    
    print(f"\n初始記憶數量: {manager.get_total_memories()}")
    manager.add_memory("我最喜歡的顏色是藍色。")
    manager.add_memory("我住在台北。")
    
    query = "你喜歡什麼顏色？"
    print(f"\n🔍 測試搜尋: \"{query}\"")
    texts, q_embed, dists = manager.search_memories(query, k=1)
    print(f"   查詢向量 (維度: {q_embed.shape if q_embed is not None else 'N/A'}): {format_vector_snippet(q_embed) if q_embed is not None else 'N/A'}")
    if texts:
        for i, txt in enumerate(texts):
            print(f"   -> 相關記憶: \"{txt}\", L2距離: {dists[i]:.4f}")
    else:
        print("   -> 未找到相關記憶。")

    manager.reload_external_memories() # 確保 reload 也正常
    print(f"重新載入後記憶數量: {manager.get_total_memories()}")

    if os.path.exists("test_faiss.idx"): os.remove("test_faiss.idx")
    if os.path.exists("test_memories.pkl"): os.remove("test_memories.pkl")
    if os.path.exists("test_persistent_memories.txt"): os.remove("test_persistent_memories.txt")
    print("\n✅ 詳細除錯模式測試完成並清理測試檔案。")