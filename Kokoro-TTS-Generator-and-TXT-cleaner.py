import sys
import subprocess
import importlib.util
import os
import re

# ==========================================
# 1. PREREQUISITE CHECK & AUTO-INSTALLER
# ==========================================
def check_and_install_requirements():
    """Checks for required modules and installs them via pip if missing."""
    
    # Warn if using Python 3.13 or higher (due to numpy/kokoro compatibility issues)
    if sys.version_info >= (3, 13):
        print("⚠️ WARNING: You are running Python 3.13 or higher.")
        print("Many AI and audio libraries (like kokoro and older numpy versions) do not have pre-built packages for 3.13 yet.")
        print("If the installation fails with a C++ / Meson error, please downgrade to Python 3.11 or 3.12.\n")

    required_packages = {
        "torch": "torch",
        "numpy": "numpy",
        "soundfile": "soundfile",
        "gradio": "gradio",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "kokoro": "kokoro"
    }

    missing_packages = []
    for module_name, pip_name in required_packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"📦 Missing required packages detected: {', '.join(missing_packages)}")
        print("⏳ Installing them globally now (this may take a few minutes)...")
        try:
            # Added --upgrade to ensure we get the best compatible versions
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", *missing_packages
            ])
            print("✅ All required packages installed successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error installing packages: {e}")
            print("If you see 'site-packages is not writeable', please close this window and run Command Prompt as Administrator.")
            print("If you see C++ build errors, you need to use Python 3.11 or 3.12.")
            sys.exit(1)

# Run the installer before importing the heavy libraries
check_and_install_requirements()

# ==========================================
# 2. IMPORTS & ENVIRONMENT SETUP
# ==========================================
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from kokoro import KPipeline

# Bypass Hugging Face token popup/warnings
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TOKEN_WARNING"] = "1"

# Global Model Variables (To avoid reloading on every click)
text_model = None
tokenizer = None
tts_pipeline = None

# ==========================================
# 3. AI MODEL LOADING & PROCESSING FUNCTIONS
# ==========================================
def load_text_model(progress=gr.Progress()):
    global text_model, tokenizer
    if text_model is None or tokenizer is None:
        progress(0, desc="Loading Qwen2.5-3B-Instruct into GPU...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Text Model loaded successfully!")

def load_tts_pipeline(lang_code, progress=gr.Progress()):
    global tts_pipeline
    if tts_pipeline is None or getattr(tts_pipeline, 'lang_code', None) != lang_code:
        progress(0, desc="Loading Kokoro TTS Model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Computing Device for TTS: {device.upper()}")
        tts_pipeline = KPipeline(lang_code=lang_code, device=device)

def process_text_with_ai(raw_text):
    system_prompt = (
        "You are an expert text editor. Your task is to clean up badly formatted text. "
        "Fix random line breaks, broken hyphenations, weird spacing, and remove inline "
        "headers, footers, or page numbers, including removing lines that just contain a single number. "
        "Preserve the original meaning and structure. "
        "Output ONLY the cleaned text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please clean the following text:\n\n{raw_text}"}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(text_model.device)
    
    generated_ids = text_model.generate(
        **model_inputs,
        max_new_tokens=2000,
        temperature=0.1,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def run_pipeline(task, input_source, file_upload, text_input, output_filename, voice, progress=gr.Progress()):
    # GET THE INPUT TEXT
    raw_input = ""
    base_name = output_filename.strip() if output_filename.strip() else "my_output"

    if input_source == "Upload .txt File":
        if file_upload is None:
            return "❌ Error: No file uploaded.", None, None
        
        with open(file_upload.name, 'r', encoding='utf-8') as f:
            raw_input = f.read()
            
        original_filename = os.path.basename(file_upload.name)
        base_name = os.path.splitext(original_filename)[0]
    else:
        raw_input = text_input

    if not raw_input.strip():
        return "⚠️ Error: No text detected. Please paste text in the box or upload a file.", None, None

    current_text = raw_input
    final_txt_path = None
    final_wav_path = None

    # TEXT CLEANER LOGIC
    if "Text Cleaner" in task or "Both" in task:
        load_text_model(progress)
        
        final_txt_path = f"{base_name}_cleaned.txt"
        
        chunk_size = 2500
        words = current_text.replace('\n', ' \n ').split(' ')
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 < chunk_size:
                current_chunk += word + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write("")

        cleaned_text_accumulated = ""
        for i, chunk in enumerate(chunks):
            progress((i + 1) / len(chunks), desc=f"Cleaning text section {i+1} of {len(chunks)}...")
            cleaned_chunk = process_text_with_ai(chunk)
            cleaned_text_accumulated += cleaned_chunk + "\n\n"
            
            with open(final_txt_path, "a", encoding="utf-8") as f:
                f.write(cleaned_chunk + "\n\n")

        current_text = cleaned_text_accumulated.strip()

    # TTS GENERATOR LOGIC
    if "TTS Generator" in task or "Both" in task:
        lang_code = voice[0]
        load_tts_pipeline(lang_code, progress)
        
        final_wav_path = f"{base_name}_cleaned.wav" if "Both" in task else f"{base_name}.wav"
        
        text_chunks = [chunk.strip() for chunk in re.split(r'(?<=[.!?\n])\s+', current_text) if chunk.strip()]
        audio_chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            progress((i + 1) / len(text_chunks), desc=f"Generating audio chunk {i+1} of {len(text_chunks)}...")
            generator = tts_pipeline(chunk_text, voice=voice, speed=1.0)
            for graphemes, phonemes, audio in generator:
                audio_chunks.append(audio)

        if audio_chunks:
            progress(1.0, desc="Merging audio files...")
            final_audio = np.concatenate(audio_chunks)
            sf.write(final_wav_path, final_audio, 24000)
        else:
            return "❌ Error: Audio generation failed.", final_txt_path, None

    return current_text, final_txt_path, final_wav_path

# ==========================================
# 4. GRADIO UI LAYOUT & LAUNCH
# ==========================================
with gr.Blocks(title="AI Text Cleaner & TTS Generator") as demo:
    gr.Markdown("# 🛠️ AI Text Cleaner & 🎙️ TTS Generator 🛠️")
    gr.Markdown("Select your Task and Input Source below to process your text or audio.")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["Text Cleaner Only", "TTS Generator Only", "Both: Clean Text then Generate TTS"],
            value="Both: Clean Text then Generate TTS",
            label="Task"
        )
        input_source_dropdown = gr.Dropdown(
            choices=["Text Box", "Upload .txt File"],
            value="Upload .txt File",
            label="Input Source"
        )
        
    with gr.Row():
        text_input = gr.Textbox(
            lines=5, 
            placeholder="Put your text here. This box is ignored if you are using a .txt file.",
            label="Input Text (If using 'Text Box')"
        )
        file_upload = gr.File(
            file_types=[".txt"], 
            label="Upload .txt File (If using 'Upload .txt File')"
        )
        
    with gr.Row():
        output_filename = gr.Textbox(
            value="my_output",
            label="Output Filename (Ignored for file uploads - uses input filename instead)"
        )
        voice_dropdown = gr.Dropdown(
            choices=["af_nova", "af_heart", "bf_emma", "am_fenrir", "bm_daniel"],
            value="af_nova",
            label="Voice (Only used if generating audio)",
            allow_custom_value=True
        )

    submit_btn = gr.Button("🚀 Run Processing", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("### 📥 Outputs")
    
    with gr.Row():
        output_text = gr.Textbox(label="Resulting Text", lines=8)
    with gr.Row():
        output_txt_file = gr.File(label="Download Cleaned Text File")
        output_audio_file = gr.Audio(label="Audio Output", type="filepath")

    # Connect UI elements to the function
    submit_btn.click(
        fn=run_pipeline,
        inputs=[task_dropdown, input_source_dropdown, file_upload, text_input, output_filename, voice_dropdown],
        outputs=[output_text, output_txt_file, output_audio_file]
    )

if __name__ == "__main__":
    print("\nStarting local server... check your browser!")
    demo.launch(inbrowser=True)