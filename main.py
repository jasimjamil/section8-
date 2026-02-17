import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

repo_id = "muhammadjasim12/incometaxcassendra"

st.set_page_config(page_title="Income Tax Chatbot", page_icon="🏛️")
st.title("🏛️ Income Tax AI Assistant")
st.caption("Ask questions about Section 8.1 of the Income Tax Assessment Act 1997")

# ── Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens  = st.slider("Max new tokens",  50, 400, 200)
    temperature = st.slider("Temperature",     0.1, 1.0, 0.7, 0.1)
    top_p       = st.slider("Top-p",           0.5, 1.0, 0.9, 0.05)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load model
@st.cache_resource(show_spinner="Loading model... please wait")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, repo_id)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
st.success("✅ Model loaded and ready!")

# ── Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input and generation
if prompt := st.chat_input("Ask about Income Tax Act Section 8.1…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            formatted = (
                f"<|system|>\nAnswer ONLY based on Section 8.1 of the Income Tax Assessment Act 1997.\n"
                f"<|user|>\n{prompt}\n<|assistant|>\n"
            )
            inputs = tokenizer(formatted, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
        
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not legal or tax advice.")
