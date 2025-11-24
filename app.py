import streamlit as st
import google.generativeai as genai
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import traceback
import re

# Page configuration
st.set_page_config(
    page_title="Duve GuestApp Localization",
    page_icon="🌍",
    layout="wide"
)

# Language mapping
LANGUAGE_OPTIONS = {
    "ar": "Arabic",
    "bg": "Bulgarian", 
    "zh": "Chinese",
    "cs": "Czech",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "th": "Thai",
    "vi": "Vietnamese"
}

# Model options
PRIMARY_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-3-pro",
    "models/gemini-2.5-pro-exp",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash"
]

FALLBACK_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash"
]

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert Hotel Copywriter and Localization Specialist.
Your task is NOT to translate words, but to convey the meaning perfectly for a Guest App in {target_lang}.

CONTEXT:
- This text is for a luxury apartment hotel.
- The tone must be: Professional, Welcoming, Premium, Helpful.
- Avoid "machine translation" feel. Use natural, idiomatic phrases.

CRITICAL RULES:
1. **NO Literal Translation:** Use natural local equivalents (e.g. HU: "Megközelítés" instead of "Saját odajutás").
2. **Keys:** You MUST rename the input keys. Replace 'en.' with '{target_lang}.' (e.g. 'en.title' -> '{target_lang}.title').
3. **Format:** Keep Markdown (**, ###) exactly as is.
4. **Placeholders:** Keep {{{{variable}}}} or _roomnumber_ unchanged.
5. **Technical:** Do not translate filenames (e.g. "1", "2")."""

# Initialize session state
if 'translation_log' not in st.session_state:
    st.session_state.translation_log = []
if 'translated_data' not in st.session_state:
    st.session_state.translated_data = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None

def log_message(message: str, level: str = "info"):
    """Add message to translation log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.translation_log.append(f"[{timestamp}] {message}")
    
def extract_english_keys(translations: Dict) -> Dict:
    """Extract only English keys (en.*) from translations object"""
    english_only = {}
    for key, value in translations.items():
        if key.startswith("en."):
            english_only[key] = value
    return english_only

def create_batches(items: Dict, batch_size: int) -> List[Dict]:
    """Split dictionary into batches"""
    items_list = list(items.items())
    batches = []
    for i in range(0, len(items_list), batch_size):
        batch_dict = dict(items_list[i:i + batch_size])
        batches.append(batch_dict)
    return batches

def translate_batch_with_retry(
    batch: Dict,
    target_lang: str,
    primary_model: str,
    fallback_model: str,
    system_prompt_template: str,
    api_key: str,
    primary_timeout: int,
    max_retries: int = 3
) -> Tuple[Dict, bool]:
    """
    Translate a batch with retry logic and fallback mechanism
    Returns: (translated_dict, success)
    """
    # Format system prompt with target language
    system_prompt = system_prompt_template.format(target_lang=LANGUAGE_OPTIONS.get(target_lang, target_lang))
    
    # Add input JSON to prompt
    full_prompt = f"""{system_prompt}

Input JSON:
{json.dumps(batch, ensure_ascii=False, indent=2)}

Output the translated JSON now:"""

    # Try primary model first
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(primary_model)
            
            response = model.generate_content(
                full_prompt,
                request_options={"timeout": primary_timeout}
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            translated = json.loads(response_text)
            return translated, True
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limiting (429)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = (2 ** attempt) * 5  # Exponential backoff
                log_message(f"⚠️ Rate limit hit for {target_lang} (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...", "warning")
                time.sleep(wait_time)
                continue
            
            # Handle server errors (500)
            elif "500" in error_msg or "internal" in error_msg.lower():
                log_message(f"⚠️ Server error for {target_lang} (attempt {attempt+1}/{max_retries}). Retrying in 3s...", "warning")
                time.sleep(3)
                continue
            
            # Handle timeout or other errors - try fallback
            elif "timeout" in error_msg.lower() or attempt == max_retries - 1:
                log_message(f"⚠️ Primary model failed for {target_lang}: {error_msg[:100]}", "warning")
                log_message(f"🔄 Switching to fallback model: {fallback_model}", "info")
                
                # Try fallback model with shorter timeout
                try:
                    fallback_timeout = min(60, primary_timeout // 2)
                    model = genai.GenerativeModel(fallback_model)
                    response = model.generate_content(
                        full_prompt,
                        request_options={"timeout": fallback_timeout}
                    )
                    response_text = response.text.strip()
                    if response_text.startswith("```"):
                        response_text = re.sub(r'^```json?\s*', '', response_text)
                        response_text = re.sub(r'\s*```$', '', response_text)
                    translated = json.loads(response_text)
                    log_message(f"✅ Fallback successful with {fallback_model}", "success")
                    return translated, True
                except Exception as fallback_error:
                    log_message(f"❌ Fallback model also failed: {str(fallback_error)[:100]}", "error")
                    return {}, False
            
            else:
                log_message(f"❌ Error translating {target_lang}: {error_msg[:100]}", "error")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return {}, False
    
    return {}, False

def translate_language(
    english_items: Dict,
    target_lang: str,
    primary_model: str,
    fallback_model: str,
    system_prompt: str,
    api_key: str,
    batch_size: int,
    timeout: int,
    progress_callback
) -> Dict:
    """Translate all items for a single language with batching"""
    batches = create_batches(english_items, batch_size)
    total_batches = len(batches)
    
    log_message(f"🔄 Starting {LANGUAGE_OPTIONS[target_lang]} ({target_lang}) - {total_batches} batch(es)", "info")
    
    all_translations = {}
    
    for idx, batch in enumerate(batches, 1):
        log_message(f"  → Processing batch {idx}/{total_batches} for {target_lang}...", "info")
        
        translated_batch, success = translate_batch_with_retry(
            batch, target_lang, primary_model, fallback_model, 
            system_prompt, api_key, timeout
        )
        
        if success:
            all_translations.update(translated_batch)
            progress_callback((idx / total_batches) * 100)
        else:
            log_message(f"  ⚠️ Batch {idx}/{total_batches} failed for {target_lang}", "warning")
    
    if len(all_translations) > 0:
        log_message(f"✅ Completed {LANGUAGE_OPTIONS[target_lang]} ({len(all_translations)} items)", "success")
    else:
        log_message(f"❌ Failed {LANGUAGE_OPTIONS[target_lang]}", "error")
    
    return all_translations

def process_translations(
    json_data: Dict,
    target_languages: List[str],
    primary_model: str,
    fallback_model: str,
    system_prompt: str,
    api_key: str,
    max_workers: int,
    batch_size: int,
    timeout: int
) -> Dict:
    """Main processing function with parallel execution"""
    
    # Extract translations object
    if "translations" not in json_data:
        log_message("❌ No 'translations' key found in JSON", "error")
        return None
    
    translations = json_data["translations"]
    
    # Clean slate: keep only English keys
    english_items = extract_english_keys(translations)
    log_message(f"📋 Found {len(english_items)} English items", "info")
    
    if len(english_items) == 0:
        log_message("❌ No English keys (en.*) found in translations", "error")
        return None
    
    # Start with English only
    result_translations = english_items.copy()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completed_languages = 0
    total_languages = len(target_languages)
    
    def update_progress(lang_progress):
        overall_progress = (completed_languages / total_languages) + (lang_progress / 100 / total_languages)
        progress_bar.progress(min(overall_progress, 1.0))
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_lang = {
            executor.submit(
                translate_language,
                english_items,
                lang,
                primary_model,
                fallback_model,
                system_prompt,
                api_key,
                batch_size,
                timeout,
                update_progress
            ): lang
            for lang in target_languages
        }
        
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                translated_items = future.result()
                result_translations.update(translated_items)
                completed_languages += 1
                status_text.text(f"Completed {completed_languages}/{total_languages} languages")
            except Exception as e:
                log_message(f"❌ Critical error for {lang}: {str(e)}", "error")
                completed_languages += 1
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Translation complete!")
    
    # Update original JSON structure
    json_data["translations"] = result_translations
    
    return json_data

# Main UI
st.title("🌍 Duve GuestApp Localization")
st.markdown("*Professional hotel content translation powered by Google Gemini AI*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Google API Key", type="password", help="Enter your Google Gemini API key")
    
    st.divider()
    
    # Primary Model selection
    primary_model = st.selectbox(
        "Primary Model",
        options=PRIMARY_MODELS,
        index=0,
        help="Main model used for translation"
    )
    
    # Fallback Model selection
    fallback_model = st.selectbox(
        "Fallback Model",
        options=FALLBACK_MODELS,
        index=0,
        help="Used if primary model fails or times out"
    )
    
    st.divider()
    
    # Language selection
    selected_languages = st.multiselect(
        "Target Languages",
        options=list(LANGUAGE_OPTIONS.keys()),
        default=list(LANGUAGE_OPTIONS.keys()),
        format_func=lambda x: f"{LANGUAGE_OPTIONS[x]} ({x})",
        help="Select languages for translation"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        max_workers = st.slider("Max Workers", min_value=1, max_value=10, value=4, help="Number of parallel translation threads")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=20, value=5, help="Items per API call")
        timeout = st.number_input("Request Timeout (seconds)", min_value=30, max_value=600, value=240, help="API call timeout")
    
    st.divider()
    st.markdown("**📊 Statistics**")
    st.metric("Selected Languages", len(selected_languages))

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 System Prompt / Translation Rules")
    system_prompt = st.text_area(
        "Edit the translation instructions below:",
        value=DEFAULT_SYSTEM_PROMPT,
        height=300,
        help="Customize how the AI translates content. Use {target_lang} as placeholder for the target language."
    )

with col2:
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Enter API key
    2. Select primary & fallback models
    3. Upload JSON file
    4. Customize system prompt
    5. Select languages
    6. Click 'Start Translation'
    7. Download result
    """)

st.divider()

uploaded_file = st.file_uploader(
    "📁 Upload Duve GuestApp JSON",
    type=['json'],
    help="Upload your exported Duve GuestApp JSON file"
)

if uploaded_file:
    try:
        json_data = json.load(uploaded_file)
        st.success(f"✅ File loaded: {uploaded_file.name}")
        st.session_state.original_filename = uploaded_file.name.replace('.json', '')
        
        # Show preview
        with st.expander("📄 File Preview"):
            if "translations" in json_data:
                english_count = sum(1 for k in json_data["translations"].keys() if k.startswith("en."))
                st.info(f"Found {english_count} English translation keys")
                st.json({k: v for k, v in list(json_data["translations"].items())[:5]})
            else:
                st.warning("No 'translations' object found in JSON")
        
        # Translation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_button = st.button(
                "🚀 Start Translation",
                type="primary",
                use_container_width=True,
                disabled=not api_key or len(selected_languages) == 0
            )
        
        if start_button:
            st.session_state.translation_log = []
            st.divider()
            
            st.subheader("📊 Translation Progress")
            
            # Log container
            log_container = st.container()
            
            # Start translation
            with st.spinner("Processing translations..."):
                result = process_translations(
                    json_data,
                    selected_languages,
                    primary_model,
                    fallback_model,
                    system_prompt,
                    api_key,
                    max_workers,
                    batch_size,
                    timeout
                )
                
                if result:
                    st.session_state.translated_data = result
                    st.success("🎉 Translation completed successfully!")
                else:
                    st.error("❌ Translation failed. Check the log below.")
            
            # Display log
            with log_container:
                with st.expander("📝 Translation Log", expanded=True):
                    for log_entry in st.session_state.translation_log:
                        st.text(log_entry)
        
        # Download button
        if st.session_state.translated_data:
            st.divider()
            
            output_filename = f"{st.session_state.original_filename}_translated_{datetime.now().strftime('%Y%m%d')}.json"
            
            json_string = json.dumps(
                st.session_state.translated_data,
                ensure_ascii=False,
                indent=2
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="⬇️ Download Translated JSON",
                    data=json_string,
                    file_name=output_filename,
                    mime="application/json",
                    type="primary",
                    use_container_width=True
                )
            
            # Show statistics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            translations = st.session_state.translated_data.get("translations", {})
            english_keys = sum(1 for k in translations.keys() if k.startswith("en."))
            total_keys = len(translations)
            translated_languages = len(set(k.split('.')[0] for k in translations.keys())) - 1  # -1 for English
            
            col1.metric("Total Keys", total_keys)
            col2.metric("English Keys", english_keys)
            col3.metric("Languages", translated_languages)
            col4.metric("Success Rate", f"{(translated_languages/len(selected_languages)*100):.0f}%")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON file.")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 <strong>Tips:</strong> Larger batch sizes are faster but may hit API limits. Start with default settings.</p>
    <p>🔒 Your API key is never stored and remains private.</p>
    <p>🎯 Customize the system prompt to fine-tune translation quality for your specific needs.</p>
</div>
""", unsafe_allow_html=True)
