from transformers import AutoTokenizer, AutoModel
import torch
from memory import MemoryManager, format_vector_snippet # 記憶管理器函數

# --- 關鍵詞設定 ---
TRIGGER_KEYWORDS = ["回憶", "記得嗎", "上次說到", "關於那件", "提醒我", "之前", "記錄","名字","愛","喜歡","討厭","是"] 
DISTANCE_THRESHOLD = 1.15 #檢索記憶向量值

def should_retrieve_memory(text, keywords):
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False

# --- ChatGLM 模型設定 ---
model_path = r"C:\temp_hf_downloads\chatglm6b" # 模型路徑

print(f"🚀 正在從本地路徑 {model_path} 載入 ChatGLM-6B 模型中...")
try:
    # 使用本地路徑載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 使用本地路徑載入模型
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    #模型移至GPU利用CUDA計算

    if torch.cuda.is_available():
        model = model.half().cuda() 
        print("ChatGLM-6B 模型已移至 GPU。")
    else:
        model = model.float() 
        print("未偵測到 CUDA，ChatGLM-6B 模型將在 CPU 上運行 (float32)。")
    model.eval()
    print("ChatGLM-6B 模型載入完成。")
except Exception as e:
    print(f"從本地路徑 {model_path} 載入 ChatGLM-6B 模型失敗: {e}")
    print("   請確保指定的本地路徑正確，且包含所有必要的模型和 Tokenizer 檔案。")
    exit()


# --- 記憶管理器設定 ---
try:
    memory_manager = MemoryManager(
        embedding_dim=384, 
        index_file="chat_faiss.idx", 
        memories_pickle_file="chat_text_memories.pkl",
        persistent_text_file="persistent_memories.txt"
    )
    print(f"✅ 記憶管理器初始化成功。目前擁有 {memory_manager.get_total_memories()} 條記憶。")
except Exception as e:
    print(f"❌ 初始化記憶管理器失敗: {e}")
    print("程式將繼續運行，但長期記憶功能可能無法使用。")
    memory_manager = None 

# --- 對話迴圈 ---
history = []
print("\n--- 對話開始 (詳細除錯模式) ---")
print("提示：輸入 '/remember <內容>' 新增記憶。")
print("提示：輸入 '/reload_memories' 從檔案重載記憶。")
print(f"提示：包含關鍵詞如 {TRIGGER_KEYWORDS[:3]}... 等時，會嘗試啟用記憶。")
print("提示：輸入 '結束' 離開。")

while True:
    input_text = input("\n👤 你想對模型說什麼：")
    
    if input_text.strip().lower() == "結束":
        print("✅ 結束對話，程式已退出。")
        if memory_manager:
            memory_manager.save_memories_on_exit() 
        break

    if input_text.lower().startswith("/remember "):
        if memory_manager:
            text_to_remember = input_text[len("/remember "):].strip()
            if text_to_remember:
                memory_manager.add_memory(text_to_remember) # add_memory 內部會打印向量片段
            else:
                print("⚠️ 記憶內容不可為空。")
        else:
            print("🛠️ 記憶管理器未成功初始化，無法新增記憶。")
        continue 
    
    if input_text.lower() == "/reload_memories":
        if memory_manager:
            memory_manager.reload_external_memories()
        else:
            print("🛠️ 記憶管理器未成功初始化，無法重載記憶。")
        continue
    # 新增：處理設定角色的指令
    if input_text.lower().startswith("/setrole "):
        persona_description = input_text[len("/setrole "):].strip()
        if persona_description:
            # 清空歷史紀錄，並設定新的角色扮演初始對話
            # "System Note:" 只是為了讓我們閱讀時更清晰，模型會把它當作使用者輸入
            # 模型的回答 "好的，我明白了..." 也是我們虛構的，目的是形成一個完整的歷史回合
            history = [
                (f"System Note: 你從現在開始將扮演以下角色：'{persona_description}'。請你完全沉浸在這個角色中，用這個角色的身份、語氣、風格和知識儲備來進行接下來所有的對話和回應。", 
                 "好的，我明白了我的新角色設定。我會努力扮演好。")
            ]
            print(f"🤖 AI 角色已成功設定為: {persona_description[:100]}...")
            print("   （對話歷史已重置以代入新角色）")
        else:
            print("⚠️ 角色描述不可為空。")
        continue # 處理完指令後，等待下一次輸入

    # 新增：處理清除角色的指令
    if input_text.lower() == "/clearrole":
        history = [] # 清空歷史紀錄，包括角色設定
        print("🤖 AI 角色已清除，恢復預設對話模式。")
        print("   （對話歷史已重置）")
        continue # 處理完指令後，等待下一次輸入

    # --- 整合長期記憶 (基於關鍵詞) ---
    retrieved_memories_text_for_prompt = ""
    
    if memory_manager and memory_manager.get_total_memories() > 0:
        # 現在我們總是嘗試檢索記憶，不再嚴格依賴 TRIGGER_KEYWORDS 作為唯一觸發
        # 關鍵詞未來可以用於調整 k 值或門檻值 (增強檢索)
        
        # 檢查是否包含關鍵詞 (可選，用於增強或特殊提示)
        keyword_detected = False
        if TRIGGER_KEYWORDS and should_retrieve_memory(input_text, TRIGGER_KEYWORDS):
            keyword_detected = True
            print(f"\n🧠 (偵測到關鍵詞，將進行記憶檢索並強化...)") # 提示用戶檢索原因
        else:
            print(f"\n🧠 (正在進行常規記憶檢索...)")

        # search_memories 返回 (texts, query_embedding, distances)
        # 可以考慮在關鍵詞觸發時取回更多的候選記憶，例如 k=5
        num_to_retrieve = 5 if keyword_detected else 3 # 關鍵詞觸發時檢索更多
        
        initial_relevant_texts, query_embedding, initial_distances = memory_manager.search_memories(input_text, k=num_to_retrieve) 
        
        if query_embedding is not None:
            print(f"    使用者輸入 \"{input_text[:50]}...\" 的向量 (維度: {query_embedding.shape}):")
            print(f"        {format_vector_snippet(query_embedding)}")
        else:
            print(f"    無法生成使用者輸入 \"{input_text[:50]}...\" 的向量。")

        genuinely_relevant_memories = [] # 儲存通過門檻的記憶文字
        if initial_relevant_texts:
            print("    初步檢索到的記憶 (將根據相似度門檻篩選)：")
            for i, mem_text in enumerate(initial_relevant_texts):
                distance = initial_distances[i]
                if distance < DISTANCE_THRESHOLD:
                    genuinely_relevant_memories.append(mem_text)
                    print(f"    ✅ \"{mem_text[:60]}...\" (L2 距離: {distance:.4f}) - 符合門檻")
                else:
                    print(f"    ❌ \"{mem_text[:60]}...\" (L2 距離: {distance:.4f}) - 未達門檻")
            
            if genuinely_relevant_memories:
                print(f"    最終選用 {len(genuinely_relevant_memories)} 條符合門檻的記憶。")
                facts = "\n".join([f"- {mem}" for mem in genuinely_relevant_memories])
                # 修改提示語，使其更通用
                retrieved_memories_text_for_prompt = (
                    "根據我們的對話，以下是一些我記錄的、與當前話題可能相關的資訊：\n"
                    f"{facts}\n\n"
                    "請參考這些資訊，並結合您的判斷來回答問題。\n"
                    "問題是：\n"
                )
            else:
                print("    ℹ️ 雖初步檢索到記憶，但無內容符合相似度門檻。")
        else:
            print("    ℹ️ 未初步檢索到任何記憶。")
            
    # 如果 genuinely_relevant_memories 為空, retrieved_memories_text_for_prompt 會是空字串
    prompt_with_memory = input_text 
    if retrieved_memories_text_for_prompt:
        prompt_with_memory = f"{retrieved_memories_text_for_prompt}\"{input_text}\""
        
    print("💬 正在生成回應...")
    try:
        response, updated_history = model.chat(tokenizer, prompt_with_memory, history=history)
        history = updated_history 
        print("\n🤖 模型回答：", response)
    except Exception as e:
        print(f"❌ 與 ChatGLM 模型對話時發生錯誤: {e}")
        print("   請檢查模型是否正確載入以及輸入格式。")