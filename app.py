import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr
import os
from openai import OpenAI
from huggingface_hub import login as hf_login

# Load environment variables
load_dotenv()


def try_load_text_file(file_path):
    """Try loading text file with multiple encodings"""
    encodings_to_try = ['utf-8', 'latin-1', 'utf-16', 'cp1252']

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            # If we get here, the encoding worked
            return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue

    raise UnicodeDecodeError(f"Failed to decode {file_path} with tried encodings: {encodings_to_try}")


# Load book data with error handling
try:
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.png",
        books["large_thumbnail"],
    )
except Exception as e:
    print(f"❌ Error loading books data: {e}")
    raise

# Load and process documents with robust encoding handling
try:
    txt_file = "tagged_description.txt"
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"'{txt_file}' not found in current directory")

    # First try the simple TextLoader with utf-8
    try:
        raw_documents = TextLoader(txt_file, encoding='utf-8').load()
    except UnicodeDecodeError:
        print("UTF-8 failed, trying alternative encodings...")
        # If that fails, use our more robust loader
        file_content, used_encoding = try_load_text_file(txt_file)
        print(f"Successfully loaded with encoding: {used_encoding}")

        # Create temporary file with correct encoding
        temp_file = "temp_tagged_description.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(file_content)

        raw_documents = TextLoader(temp_file).load()
        os.remove(temp_file)  # Clean up temporary file

    print("Successfully loaded documents, proceeding with text splitting...")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents)} document chunks")

    print("Creating Chroma vector store...")
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
    print("Vector store created successfully")

except Exception as e:
    print(f"❌ Error processing documents: {e}")
    print("Possible solutions:")
    print("1. Check if the file exists in the correct location")
    print("2. Verify the file contains valid text content")
    print("3. Try converting the file to UTF-8 encoding using a text editor")
    print("4. Check for special characters in the file")
    raise


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    try:
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category]

        if tone and tone != "All":
            tone_mapping = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)

        return book_recs.head(final_top_k)
    except Exception as e:
        print(f"⚠️ Error in recommendation: {e}")
        return pd.DataFrame()


def recommend_books(query: str, category: str, tone: str):
    try:
        if not query.strip():
            return [("cover-not-found.png", "Please enter a book description")]

        recommendations = retrieve_semantic_recommendations(query, category, tone)

        if recommendations.empty:
            return [("cover-not-found.png", "No books found matching your criteria")]

        results = []
        for _, row in recommendations.iterrows():
            description = row.get("description", "No description available")
            truncated_desc = " ".join(description.split()[:30]) + "..." if description else "No description"

            authors = row.get("authors", "Unknown author")
            authors_split = authors.split(";")

            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors

            title = row.get("title", "Untitled")
            caption = f"{title} by {authors_str}: {truncated_desc}"
            results.append((row["large_thumbnail"], caption))

        return results
    except Exception as e:
        print(f"⚠️ Error in processing recommendations: {e}")
        return [("cover-not-found.png", "An error occurred while processing your request")]


# Prepare UI options
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create responsive interface
# [Previous imports and data loading code remains the same until the Gradio interface section]

# Create responsive interface with improved gallery navigation
with gr.Blocks(theme=gr.themes.Soft(), title="Find your read") as dashboard:
    gr.Markdown("""
    # 📚  Find your read
    *Find your next favorite book based on content and mood*
    """)

    with gr.Row():
        with gr.Column(scale=4):
            user_query = gr.Textbox(
                label="Describe what you're looking for",
                placeholder="e.g., A fantasy adventure about friendship and magic",
                max_lines=3
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Category",
                value="All"
            )
        with gr.Column(scale=1):
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Mood",
                value="All"
            )

    submit_btn = gr.Button("Find Recommendations", variant="primary")

    gr.Markdown("## Recommended Books")

    # Create a row for navigation controls and gallery
    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous", variant="secondary", visible=False)
        next_btn = gr.Button("➡️ Next", variant="secondary", visible=False)

    # Store current page index
    current_page = gr.State(0)
    # Store all recommendations
    all_recommendations = gr.State([])

    # Gallery with fixed grid layout
    gallery = gr.Gallery(
        label="Results",
        columns=4,  # Fixed 4 columns for better grid layout
        rows=2,
        object_fit="cover",
        height="auto",
        preview=False  # Disable the lightbox preview
    )


    # Function to show current page of recommendations
    def show_page(recommendations, page):
        if not recommendations:
            return [], 0, False, False

        items_per_page = 8  # 4 columns x 2 rows
        total_pages = (len(recommendations) + items_per_page - 1) // items_per_page
        start_idx = page * items_per_page
        end_idx = min((page + 1) * items_per_page, len(recommendations))

        show_prev = page > 0
        show_next = page < total_pages - 1

        return recommendations[start_idx:end_idx], page, show_prev, show_next


    # Handle recommendation search
    def get_recommendations(query, category, tone):
        results = recommend_books(query, category, tone)
        return results, 0, False, len(results) > 8


    # Navigation functions
    def next_page(page, recommendations):
        items_per_page = 8
        total_pages = (len(recommendations) + items_per_page - 1) // items_per_page
        new_page = min(page + 1, total_pages - 1)
        return *show_page(recommendations, new_page), new_page


    def prev_page(page, recommendations):
        new_page = max(page - 1, 0)
        return *show_page(recommendations, new_page), new_page


    # Connect the buttons
    submit_btn.click(
        fn=get_recommendations,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[all_recommendations, current_page, prev_btn, next_btn]
    ).then(
        fn=show_page,
        inputs=[all_recommendations, current_page],
        outputs=[gallery, current_page, prev_btn, next_btn]
    )

    next_btn.click(
        fn=next_page,
        inputs=[current_page, all_recommendations],
        outputs=[gallery, current_page, prev_btn, next_btn, current_page]
    )

    prev_btn.click(
        fn=prev_page,
        inputs=[current_page, all_recommendations],
        outputs=[gallery, current_page, prev_btn, next_btn, current_page]
    )

if __name__ == "__main__":
    dashboard.launch()