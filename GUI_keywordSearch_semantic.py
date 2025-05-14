import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from fuzzywuzzy import fuzz
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch

DEFAULT_KEYWORD = 'ਵੋਟ'
threshold = 80

# Global state
current_file_path = None
current_content = ''
current_original = ''
current_exact_spans = []
current_fuzzy_spans = []
current_keyword = DEFAULT_KEYWORD

def process_file(content, keyword):
    pattern = re.compile(re.escape(keyword), re.UNICODE)
    exact_matches = [(m.start(), m.end()) for m in pattern.finditer(content)]

    words = set(re.findall(r'\w+', content, flags=re.UNICODE))
    fuzzy_matches = [word for word in words if word != keyword and fuzz.partial_ratio(word, keyword) >= threshold]
    fuzzy_matches.sort(key=lambda x: -len(x))

    fuzzy_match_spans = []
    for word in fuzzy_matches:
        for match in re.finditer(rf'\b{re.escape(word)}\b', content, flags=re.UNICODE):
            start, end = match.start(), match.end()
            if not any(es <= start < ee or es < end <= ee for es, ee in exact_matches):
                fuzzy_match_spans.append((start, end))

    return len(exact_matches), len(fuzzy_match_spans), exact_matches, fuzzy_match_spans

def highlight_matches(text_widget, content, exact_spans, fuzzy_spans):
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, content)

    text_widget.tag_delete("exact")
    text_widget.tag_delete("fuzzy")
    text_widget.tag_config("exact", background="#c3f7c6")
    text_widget.tag_config("fuzzy", background="#fff5b1")

    for start, end in exact_spans:
        text_widget.tag_add("exact", f"1.0+{start}c", f"1.0+{end}c")
    for start, end in fuzzy_spans:
        text_widget.tag_add("fuzzy", f"1.0+{start}c", f"1.0+{end}c")

    text_widget.config(state=tk.DISABLED)

def save_file_as_text(content, exact_spans, fuzzy_spans, output_file_path):
    span_labels = [('exact', s, e) for s, e in exact_spans] + [('fuzzy', s, e) for s, e in fuzzy_spans]
    span_labels.sort(key=lambda x: x[1])

    result = ''
    cursor = 0
    for typ, start, end in span_labels:
        if start >= cursor:
            result += content[cursor:start]
            word = content[start:end]
            if typ == 'exact':
                result += f"_{word}_"
            else:
                result += f"**{word}**"
            cursor = end
    result += content[cursor:]

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(result)

def save_output_file():
    if not current_content or not current_file_path:
        return

    base_name = os.path.splitext(os.path.basename(current_file_path))[0]
    output_file_name = f"{base_name}_text.txt"
    output_dir = os.path.dirname(current_file_path)
    output_file_path = os.path.join(output_dir, output_file_name)

    save_file_as_text(current_content, current_exact_spans, current_fuzzy_spans, output_file_path)
    messagebox.showinfo("File Saved", f"Output saved as:\n{output_file_path}")


def rerun_search(new_keywords=None):
    global current_keyword, current_exact_spans, current_fuzzy_spans

    if new_keywords:
        current_keyword = new_keywords.strip()
    if not current_keyword:
        messagebox.showwarning("Invalid Input", "Please enter a keyword.")
        return

    keywords = [kw.strip() for kw in current_keyword.split(",") if kw.strip()]
    if not keywords:
        messagebox.showwarning("Invalid Input", "Please enter at least one keyword.")
        return

    total_exact_spans = []
    total_fuzzy_spans = []
    result_texts = []

    for kw in keywords:
        exact_count, fuzzy_count, exact_spans, fuzzy_spans = process_file(current_content, kw)
        total_exact_spans.extend(exact_spans)
        total_fuzzy_spans.extend(fuzzy_spans)
        result_texts.append(f"'{kw}': {exact_count} Exact, {fuzzy_count} Fuzzy")

    current_exact_spans = total_exact_spans
    current_fuzzy_spans = total_fuzzy_spans

    # Update result display
    update_keyword_result_boxes(result_texts)
    highlight_matches(output_box, current_content, total_exact_spans, total_fuzzy_spans)


    exact_count, fuzzy_count, exact_spans, fuzzy_spans = process_file(current_content, current_keyword)
    current_exact_spans = exact_spans
    current_fuzzy_spans = fuzzy_spans

    exact_label_var.set(f"{exact_count} Exact Matches")
    fuzzy_label_var.set(f"{fuzzy_count} Fuzzy Matches")
    highlight_matches(output_box, current_content, exact_spans, fuzzy_spans)

def show_output_page(file_path):
    global current_file_path, current_content, current_original
    global current_keyword, current_exact_spans, current_fuzzy_spans

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        messagebox.showerror("File Error", f"Failed to read file: {e}")
        return

    current_file_path = file_path
    current_content = content
    current_original = content
    current_keyword = DEFAULT_KEYWORD

    first_frame.pack_forget()
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    keyword_entry_var.set(current_keyword)
    original_box.config(state=tk.NORMAL)
    original_box.delete(1.0, tk.END)
    original_box.insert(tk.END, current_original)
    original_box.config(state=tk.DISABLED)

    rerun_search(current_keyword)

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select Punjabi Text File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if file_path:
        show_output_page(file_path)

def go_back():
    canvas_frame.pack_forget()
    first_frame.pack(padx=10, pady=10)

# === Main Window ===
root = tk.Tk()
root.title("Punjabi Keyword Highlighter")
root.state("zoomed")  # Opens maximized on Windows
# OR use below for cross-platform max fit:
root.attributes("-fullscreen", False)
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 12), padding=6)
style.configure("TLabel", font=("Segoe UI", 12))
style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))

# === First Page ===
first_frame = ttk.Frame(root)
first_frame.place(relx=0.5, rely=0.5, anchor="center")
ttk.Label(first_frame, text="📄 Punjabi Keyword Highlighter", style="Header.TLabel").pack(pady=(0, 20))
ttk.Button(first_frame, text="Select File", command=open_file).pack()

# === Second Page (Canvas with Scroll) ===
canvas_frame = ttk.Frame(root)
canvas = tk.Canvas(canvas_frame)
scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# === Keyword Input & Search ===
search_frame = ttk.Frame(scrollable_frame, padding=10)
search_frame.pack(fill=tk.X)
keyword_entry_var = tk.StringVar()
ttk.Label(search_frame, text="🔍 Enter Keyword:").pack(side=tk.LEFT, padx=5)
keyword_entry = ttk.Entry(search_frame, textvariable=keyword_entry_var, width=20)
keyword_entry.pack(side=tk.LEFT, padx=5)
ttk.Button(search_frame, text="Search", command=lambda: rerun_search(keyword_entry_var.get())).pack(side=tk.LEFT, padx=5)

# === Count Display (Each Keyword in its Own Box) ===
info_frame = ttk.Frame(scrollable_frame, padding=10)
info_frame.pack()
keyword_result_boxes = []  # Store labels so we can update them later

def update_keyword_result_boxes(results):
    # Clear existing boxes
    for widget in keyword_result_boxes:
        widget.destroy()
    keyword_result_boxes.clear()

    # Create a new box for each keyword result
    for result in results:
        lbl = ttk.Label(info_frame, text=result, background="#e6f7ff", relief="ridge",
                        padding=8, anchor="center", width=30)
        lbl.pack(side=tk.LEFT, padx=5, pady=5)
        keyword_result_boxes.append(lbl)


# === Dual Frame: Original & Output ===
dual_frame = ttk.Frame(scrollable_frame, padding=10)
dual_frame.pack(fill=tk.BOTH, expand=True)

# Original File Viewer
original_container = ttk.Frame(dual_frame)
original_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
ttk.Label(original_container, text="📄 Original File Content", style="Header.TLabel").pack()
original_box = tk.Text(original_container, wrap=tk.WORD, font=("Segoe UI", 12), background="#f3f3f3", height=25)
original_box.pack(fill=tk.BOTH, expand=True)
ttk.Scrollbar(original_container, command=original_box.yview).pack(side=tk.RIGHT, fill=tk.Y)
original_box.config(yscrollcommand=lambda *args: original_box.yview(*args), state=tk.DISABLED)

# Highlighted Output Viewer
output_container = ttk.Frame(dual_frame)
output_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
ttk.Label(output_container, text="🔍 Highlighted Output", style="Header.TLabel").pack()
output_box = tk.Text(output_container, wrap=tk.WORD, font=("Segoe UI", 12), height=25)
output_box.pack(fill=tk.BOTH, expand=True)
ttk.Scrollbar(output_container, command=output_box.yview).pack(side=tk.RIGHT, fill=tk.Y)
output_box.config(yscrollcommand=lambda *args: output_box.yview(*args), state=tk.DISABLED)


# === Semantic Search Frame (3rd screen) ===
semantic_frame = ttk.Frame(root)

def open_semantic_page():
    canvas_frame.pack_forget()  # 2nd screen
    semantic_frame.pack(fill=tk.BOTH, expand=True)

def go_back_from_semantic():
    semantic_frame.pack_forget()
    canvas_frame.pack(fill=tk.BOTH, expand=True)

# Semantic input UI
semantic_input_frame = ttk.Frame(semantic_frame, padding=20)
semantic_input_frame.pack()
ttk.Label(semantic_input_frame, text="🔍 Enter comma-separated keywords for semantic search:", style="Header.TLabel").pack(pady=10)

semantic_entry_var = tk.StringVar(value="ਪਿਸਤੌਲ, ਸਪਲਾਇਰ, ਡ੍ਰੌਪ-ਆਫ਼, ਹਥਿਆਰ, ਦਵਾਈ, ਰਾਈਫਲ, ਡ੍ਰੌਪ, ਕੈਦੀ")
semantic_entry = ttk.Entry(semantic_input_frame, textvariable=semantic_entry_var, width=70)
semantic_entry.pack(pady=5)

# Output area
semantic_output_frame = ttk.Frame(semantic_frame)
semantic_output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

scrollbar = ttk.Scrollbar(semantic_output_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

semantic_output = tk.Text(semantic_output_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                          font=("Segoe UI", 12), height=20, background="#f7faff")
semantic_output.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=semantic_output.yview)

semantic_output.config(state=tk.DISABLED)

def run_semantic_search():
    semantic_output.config(state=tk.NORMAL)
    semantic_output.delete(1.0, tk.END)

    keywords_input = semantic_entry_var.get().strip()
    if not keywords_input:
        messagebox.showwarning("Input Error", "Please enter at least one keyword.")
        return

    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
    if not keywords:
        messagebox.showwarning("Input Error", "No valid keywords provided.")
        return

    try:
        semantic_output.insert(tk.END, "🔄 Loading multilingual semantic model...\n")
        root.update_idletasks()
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        conversation = current_content
        conversation_embedding = model.encode(conversation, convert_to_tensor=True)
        keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

        similarity_scores = util.cos_sim(conversation_embedding, keyword_embeddings)

        semantic_output.insert(tk.END, "\n✅ Similarity Scores + Match Counts:\n")
        for i, keyword in enumerate(keywords):
            score = similarity_scores[0][i].item()

            # Count exact and fuzzy matches
            exact_count, fuzzy_count, _, _ = process_file(current_content, keyword)

            # Display result
            semantic_output.insert(
                tk.END,
                f"• '{keyword}' → {score:.4f} (Exact: {exact_count}, Fuzzy: {fuzzy_count})\n"
            )
    except Exception as e:
        semantic_output.insert(tk.END, f"\n❌ Error: {str(e)}\n")

    # Add final bolded note
    semantic_output.tag_configure("bold", font=("Segoe UI", 12, "bold"))
    semantic_output.insert(
        tk.END,
        "\n\n📌 In general, a score above 0.3 indicates a good match of the keyword in the selected script. "
        "You can check the number of exact and fuzzy matches in brackets to judge Suspiciousness of the script.",
        "bold"
    )
    
    semantic_output.config(state=tk.DISABLED)


# Buttons at bottom of semantic screen
semantic_button_frame = ttk.Frame(semantic_frame, padding=10)
semantic_button_frame.pack()

ttk.Button(semantic_button_frame, text="🚀 Run Semantic Search", command=run_semantic_search).pack(side=tk.LEFT, padx=10)
ttk.Button(semantic_button_frame, text="⬅ Back", command=go_back_from_semantic).pack(side=tk.LEFT, padx=10)


# === Buttons ===
button_frame = ttk.Frame(scrollable_frame, padding=10)
button_frame.pack()
ttk.Button(button_frame, text="💾 Save Output", command=save_output_file).pack(side=tk.LEFT, padx=10)
ttk.Button(button_frame, text="⬅ Back", command=go_back).pack(side=tk.LEFT, padx=10)
ttk.Button(button_frame, text="🔎 Semantic Search", command=open_semantic_page).pack(side=tk.LEFT, padx=10)

root.mainloop()

