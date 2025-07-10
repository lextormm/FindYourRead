Find Your Read
Find Your Read is a Generative AI-powered book recommendation engine that suggests books based on user-inputted descriptions, moods, genres, and preferences. Built using OpenAI embeddings, LangChain and ChromaDB, this project brings personalized book discovery to life.

Features

1. Personalized Book Recommendations: Users describe the kind of book they want, and the system suggests the best matches.

2. Flexible Filtering: Choose from genres (fiction, non-fiction, children's books) and moods (happy, sad, thrilling, etc).

3. Natural Language Understanding: Uses OpenAI and Hugging Face models for semantic search and text understanding.

4. Lightweight Frontend: Built using Gradio for quick and simple user interaction.


Tech Stack

1. Frontend: Gradio

2. AI/ML: OpenAI API, Hugging Face, LangChain

3. Database: ChromaDB for vector storage

4. Dataset: Cleaned Kaggle book dataset


How to Run Locally

1. Clone the repository.

2. Set up .env with your OpenAI, Hugging Face API keys.

3. Install required Python packages: pip install -r requirements.txt

4. Run the app: python app.py


Future Improvements

1. Integration with real-time book databases or APIs (Goodreads, Google Books)

2. Personalized user profiles and reading history

3. Mobile-friendly deployment
