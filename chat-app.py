import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.title("KSS-history-chat")
st.set_page_config(page_title="Amerikanische Geschichte", page_icon="🧠", layout="centered", initial_sidebar_state="auto", menu_items=None)

# Abfrage des OpenAI API-Schlüssels mithilfe eines Eingabefelds
openai_key = st.text_input("Bitte gib deinen OpenAI API-Schlüssel ein:", type="password")

# Wenn kein Schlüssel eingegeben wurde, wird das Programm angehalten.
if not openai_key:
    st.warning("Bitte gib deinen OpenAI API-Schlüssel ein, um fortzufahren.")
    st.stop()

openai.api_key = openai_key

# st.title("Amerikanische Geschichte")

if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Stell mir eine Frage!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Die Daten werden geladen. – Das kann etwas dauern..."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Du bist ein Experte für amerikanische Geschichte und deine Aufgabe ist es, Fragen zur amerikanischen Geschichte zu beantworten. Gehe davon aus, dass sich alle Fragen auf die frühen Siedlungen in Nordamerika, den Unabhängigkeitskampf, die Entwicklung der politischen Institutionen und Gesellschaft sowie die Ereignisse der Amerikanischen Revolution beziehen. Formuliere die Antworten kurz und in einfachen Sätzen und stütze dich auf die historischen Fakten - halluziniere nicht!"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Deine Frage"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
