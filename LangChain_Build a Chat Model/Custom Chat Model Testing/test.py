# 假設檔案結構如下：
# .
# ├─ chat_parrot_link.py  # 這裡面定義了您給的 ChatParrotLink 類別
# └─ test_parrot.py       # 您想要測試的程式

from langchain_core.messages import HumanMessage
from custom_chat_model import ChatParrotLink  # 依據實際檔案位置做 import

def main():
    # 初始化自訂模組，設定回音長度 parrot_buffer_length=5
    model = ChatParrotLink(
        model="my-parrot-model",    # model_name => alias="model"
        parrot_buffer_length=5
    )

    # 建立一個 HumanMessage
    user_input = "Hello, ChatParrot!"
    human_message = HumanMessage(content=user_input)

    # 呼叫 .invoke() 方法取得回應
    response = model.invoke([human_message])

    # ChatResult 會包含 generations，是一個 list
    # 每個 generation 裏都有一個 AIMessage
    # 若只需要單一回答，可以直接使用 [0] 取出

    print("User Input: ", user_input)
    print("Parrot Output:", response.content)

if __name__ == "__main__":
    main()
