import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # (openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Error handling for no URLs provided
if process_url_clicked:
    if not any(urls):  # Check if at least one URL is provided
        st.error("No URLs provided. Please provide at least one valid URL.")
    else:
        try:
            # load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()

            if not data:
                st.error("Failed to load data from the provided URLs. Please check the URLs and try again.")
            else:
                # split data
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)
                
                if not docs:
                    st.error("No documents to process after text splitting.")
                else:
                    # create embeddings and save it to FAISS index
                    embeddings = OpenAIEmbeddings()
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)

                    # Save the FAISS index to a pickle file
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_openai, f)

                    main_placeholder.success("Data processing complete. You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred while processing the URLs: {e}")

# Error handling for no query provided
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                
                # Handling possible retrieval errors
                try:
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                    
                    # Display the result
                    if result:
                        st.header("Answer")
                        st.write(result["answer"])

                        # Display sources, if available
                        sources = result.get("sources", "")
                        if sources:
                            st.subheader("Sources:")
                            sources_list = sources.split("\n")  # Split the sources by newline
                            for source in sources_list:
                                st.write(source)
                    else:
                        st.error("No result found for your query. Please try again.")

                except Exception as e:
                    st.error(f"An error occurred during the retrieval process: {e}")

        except FileNotFoundError:
            st.error("The FAISS index file was not found. Please process the URLs first.")
        except pickle.UnpicklingError:
            st.error("Failed to load the FAISS index from the pickle file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("FAISS index file not found. Please process URLs first.")

else:
    if st.button("Submit Question"):
        st.error("Please enter a question to submit.") 
