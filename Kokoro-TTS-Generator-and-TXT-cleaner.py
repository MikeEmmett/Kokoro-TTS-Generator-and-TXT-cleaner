import os
import sys
import subprocess
import importlib.util
import shutil
import re

# --- PYTHON VERSION CHECK ---
if sys.version_info < (3, 8):
    print("❌ Error: Python 3.8 or higher is required.")
    sys.exit(1)

# --- AUTO DEPENDENCY INSTALLER ---
def check_and_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
        
    if importlib.util.find_spec(import_name) is None:
        print(f"📦 '{package_name}' is missing. Installing it now...")
        try:
            # sys.executable ensures we use the pip associated with the current Python environment
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
            print(f"✅ Successfully installed {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}. Please install it manually.")
            sys.exit(1)

print("🔍 Checking system dependencies...")
check_and_install("gradio")
check_and_install("transformers")
check_and_install("accelerate")
check_and_install("torch")
check_and_install("soundfile")
check_and_install("kokoro")
check_and_install("numpy")
print("✅ All dependencies are satisfied.\n")

# Now it is safe to import gradio
import gradio as gr

# --- BYPASS HUGGING FACE TOKEN POPUP ---
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TOKEN_WARNING"] = "1"

# --- LAZY LOADING GLOBALS ---
# We keep these global so they only load into VRAM once per session
text_model = None
text_tokenizer = None
tts_pipeline = None

def check_gpu():
    import torch
    if torch.cuda.is_available():
        return f"✅ GPU Available: {torch.cuda.get_device_name(0)}"
    return "⚠️ WARNING: GPU not detected. Generation will be slow (CPU only)."

def load_text_model():
    global text_model, text_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if text_model is None or text_tokenizer is None:
        print("⏳ Loading Qwen2.5-3B-Instruct into GPU...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Text Model loaded successfully!")

def load_tts_model(voice):
    global tts_pipeline
    import torch
    from kokoro import KPipeline
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lang_code = voice[0] # Usually 'a' or 'b'
    
    # Reload if pipeline isn't loaded or language code changed
    if tts_pipeline is None or tts_pipeline.lang_code != lang_code:
        print(f"⏳ Loading Kokoro-82M model ({lang_code}) onto {device.upper()}...")
        tts_pipeline = KPipeline(lang_code=lang_code, device=device)
        print("✅ TTS Model loaded successfully!")

def process_pipeline(task, input_source, uploaded_file, text_input, output_filename, zip_password, voice, system_prompt):
    import torch
    import numpy as np
    import soundfile as sf
    
    # 1. Setup Output Directory
    os.makedirs("outputs", exist_ok=True)
    logs = []
    output_files_for_download = []
    
    def log(msg):
        print(msg)
        logs.append(msg)
        
    log("🚀 Starting processing pipeline...")
    
    # 2. Get Input Text
    raw_input = ""
    base_name = output_filename if output_filename.strip() else "Processed_File"
    
    if input_source == "Upload .txt File":
        if not uploaded_file:
            return "\n".join(logs) + "\n❌ Error: No file uploaded.", None, None
        
        with open(uploaded_file, 'r', encoding='utf-8') as f:
            raw_input = f.read()
        base_name = os.path.splitext(os.path.basename(uploaded_file))[0]
        log(f"✅ Loaded file successfully.")
    else:
        raw_input = text_input

    if not raw_input.strip():
        return "\n".join(logs) + "\n⚠️ Error: No text detected.", None, None

    current_text = raw_input
    final_txt_path = os.path.join("outputs", f"{base_name}_cleaned.txt")
    final_wav_path = os.path.join("outputs", f"{base_name}_cleaned.wav" if "Both" in task else f"{base_name}.wav")
    audio_output_file = None

    # 3. TEXT CLEANER LOGIC
    if "Text Cleaner" in task or "Both" in task:
        log("\n🖨️ STARTING AI TEXT CLEANER...")
        load_text_model()
        
        def process_text_with_ai(raw_text):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please clean the following text:\n\n{raw_text}"}
            ]
            formatted_prompt = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = text_tokenizer([formatted_prompt], return_tensors="pt").to(text_model.device)
            generated_ids = text_model.generate(
                **model_inputs,
                max_new_tokens=2000,
                temperature=0.1,
                do_sample=True,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            return text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Chunking
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

        log(f"🧩 Document safely split into {len(chunks)} manageable chunks.")

        # Clear output file
        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write("")

        for i, chunk in enumerate(chunks):
            log(f"⏳ Cleaning section {i+1} of {len(chunks)}...")
            cleaned_chunk = process_text_with_ai(chunk)
            with open(final_txt_path, "a", encoding="utf-8") as f:
                f.write(cleaned_chunk + "\n\n")

        log(f"🎉 Cleaned text saved to: {final_txt_path}")
        
        # Reload text for TTS step
        with open(final_txt_path, "r", encoding="utf-8") as f:
            current_text = f.read()
            
        output_files_for_download.append(final_txt_path)

    # 4. TTS GENERATOR LOGIC
    if "TTS Generator" in task or "Both" in task:
        log("\n🎙️ STARTING KOKORO TTS GENERATOR...")
        load_tts_model(voice)
        
        log(f"Generating speech for voice: {voice}...")
        text_chunks = [chunk.strip() for chunk in re.split(r'(?<=[.!?\n])\s+', current_text) if chunk.strip()]
        total_chunks = len(text_chunks)
        audio_chunks = []

        for i, chunk_text in enumerate(text_chunks):
            generator = tts_pipeline(chunk_text, voice=voice, speed=1.0)
            for graphemes, phonemes, audio in generator:
                audio_chunks.append(audio)
            log(f"  -> Processed audio chunk {i+1} out of {total_chunks}...")

        if audio_chunks:
            log("Merging audio chunks...")
            final_audio = np.concatenate(audio_chunks)
            sf.write(final_wav_path, final_audio, 24000)
            log(f"✅ Successfully created audio: {final_wav_path}")
            audio_output_file = final_wav_path
            output_files_for_download.append(final_wav_path)
        else:
            log("❌ Error: No audio was generated.")

    # 5. ENCRYPTION LOGIC (7-zip)
    if zip_password.strip() and output_files_for_download:
        if shutil.which("7z") is None:
            log("⚠️ Warning: 7-zip is not installed on this system. Skipping encryption.")
        else:
            log(f"🔒 Encrypting output files...")
            encrypted_files = []
            for file_path in output_files_for_download:
                archive_name = f"{file_path}.7z"
                subprocess.run(["7z", "a", f"-p{zip_password}", "-mhe=on", archive_name, file_path], stdout=subprocess.DEVNULL)
                encrypted_files.append(archive_name)
                log(f"✅ Encrypted {file_path} -> {archive_name}")
            
            # Offer the encrypted files for download instead of the raw ones
            output_files_for_download = encrypted_files

    log("\n✨ All tasks completed successfully!")
    return "\n".join(logs), audio_output_file, output_files_for_download


# ==========================================
# GRADIO WEB UI SETUP
# ==========================================
with gr.Blocks(title="AI Text Cleaner & TTS Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ AI Text Cleaner & 🎙️ TTS Generator")
    gr.Markdown(f"**System Status:** {check_gpu()}")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Task & Source")
            task_dropdown = gr.Radio(
                choices=["Text Cleaner Only", "TTS Generator Only", "Both: Clean Text then Generate TTS"],
                value="Both: Clean Text then Generate TTS",
                label="Select your Task"
            )
            input_source = gr.Radio(
                choices=["Text Box", "Upload .txt File"],
                value="Upload .txt File",
                label="Input Source"
            )
            
            file_upload = gr.File(label="Upload .txt File", file_types=[".txt"], visible=True)
            text_input = gr.Textbox(
                label="Text Input", 
                lines=5, 
                placeholder="Paste your text here...",
                visible=False
            )
            
            # Dynamic visibility for input types
            def toggle_input(choice):
                if choice == "Upload .txt File":
                    return gr.update(visible=True), gr.update(visible=False)
                return gr.update(visible=False), gr.update(visible=True)
            
            input_source.change(toggle_input, inputs=input_source, outputs=[file_upload, text_input])

        with gr.Column():
            gr.Markdown("### 2. General Settings")
            output_filename = gr.Textbox(
                label="Output Filename (Optional)", 
                value="Processed_File",
                info="Used if inputting via Text Box. File uploads will inherit the original filename."
            )
            zip_password = gr.Textbox(
                label="Zip Password (Optional)", 
                type="password",
                info="Requires 7-zip installed on your OS. Leave blank for no encryption."
            )
            
            gr.Markdown("### 3. TTS Settings")
            voice_dropdown = gr.Dropdown(
                choices=["af_nova", "af_heart", "bf_emma", "am_fenrir", "bm_daniel"],
                value="af_nova",
                label="Voice Selection",
                allow_custom_value=True
            )

    with gr.Accordion("🛠️ Advanced Settings", open=False):
        system_prompt = gr.Textbox(
            label="System Prompt for Text Cleaner",
            lines=4,
            value="You are an expert audio-text preparer. Your task is to clean this text so it reads smoothly for Text-to-Speech processing. 1. Remove random line breaks to reconstruct proper flowing paragraphs. 2. Fix broken hyphenations (e.g., 'para- graph' becomes 'paragraph'). 3. Normalize spacing by removing extra spaces or tabs. 4. Delete inline headers, footers, page numbers, and stray isolated numbers. 5. DO NOT rewrite, summarize, or change the author's original words. Output ONLY the cleaned text with no conversational filler."
        )

    submit_btn = gr.Button("🚀 Process Now", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("### 📥 Outputs")
    with gr.Row():
        with gr.Column(scale=2):
            status_log = gr.Textbox(label="Execution Log", lines=10, interactive=False)
        with gr.Column(scale=1):
            audio_player = gr.Audio(label="Generated Audio", interactive=False)
            file_downloader = gr.File(label="Download Files")

    # Connect UI to logic
    submit_btn.click(
        fn=process_pipeline,
        inputs=[
            task_dropdown, input_source, file_upload, text_input, 
            output_filename, zip_password, voice_dropdown, system_prompt
        ],
        outputs=[status_log, audio_player, file_downloader]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)