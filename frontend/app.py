import streamlit as st
import requests

st.title("电影推荐与问答系统")

# 用户输入问题
query = st.text_input("请输入您的问题：")

if st.button("提交"):
    if query:
        # 调用后端 API
        response = requests.post("http://localhost:8000/query", json={"query": query})
        if response.status_code == 200:
            st.write("回答：", response.json()["response"])
        else:
            st.write("请求失败，请稍后再试。")
    else:
        st.write("请输入有效的问题。")