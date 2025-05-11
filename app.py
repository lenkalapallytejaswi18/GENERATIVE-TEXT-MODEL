import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Generative Text Model", layout="centered")

# 🔄 Cache model loading
@st.cache_resource
def load_model():
#Loads the GPT-2 model and its tokenizer from Hugging Face:


    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# 🔄 Text generation function
def generate_paragraph(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    #Encodes the user input (prompt) into tokens.
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length, #total length of output tokens
            do_sample=True,
            temperature=temperature,
            top_k=top_k,#considers top-k probable words at each step
            top_p=top_p,# Nucleus sampling chooses from the top words until they add up to probability p
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 🌐 Streamlit UI
st.title("🧠 Topic-Based Text Generator")
st.markdown("Generate coherent paragraphs using GPT-2 on any topic.")

# ✏️ User input
topic = st.text_input("🔤 Enter a topic:", "The future of AI") # for topic 
max_len = st.slider("📏 Max paragraph length:", 50, 500, 200) # for para
temp = st.slider("🔥 Creativity (temperature):", 0.2, 1.0, 0.7)

# ▶️ Generate
if st.button("Generate Text"): # calls the generate paragraph
    with st.spinner("Generating..."):
        result = generate_paragraph(topic, max_length=max_len, temperature=temp)
        st.subheader("📝 Generated Paragraph:")
        st.write(result) # for displaying results 
