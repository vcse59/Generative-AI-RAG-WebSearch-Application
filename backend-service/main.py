from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import httpx
from fastapi.middleware.cors import CORSMiddleware
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from enum import Enum, auto

# Load environment variables from .env file
load_dotenv()


# download ollama models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download models at startup."""
    await download_ollama_models()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="RAG API using websearch",
    description="Retrieval-Augmented Generation (RAG) API using web search as knowledge source.",
    version="1.0.0",
    lifespan=lifespan
)

# Define CORS configuration
origins = [
    "*",  # To allow all origins (use with caution in production)
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins or specify a list
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Input Schema
class RequestModel(BaseModel):
    prompt: str

# Response Schema
class ResponseModel(BaseModel):
    response: str
    request: str

# Ollama API URL
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL")  # Ollama API URL, can be overridden by setting shell variable

# Specify the absolute or relative path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../config/.env')

# Load the .env file from the given path
load_dotenv(dotenv_path=dotenv_path)

# Access variables
OLLAMA_LLM_MODEL_NAME   =   os.environ.get("OLLAMA_LLM_MODEL_NAME")

# Define search function (using a simple search engine API)
def search_web(query: str):
    try:
        # Properly encode the query to handle UTF-8 characters
        encoded_query = quote_plus(query)
        print(f"Encoded Query: {encoded_query}")  # Debug: check encoded query
        
        # Use DuckDuckGo API for demo purposes
        search_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        print(f"Search URL: {search_url}")  # Debug: check the full URL
        
        response = requests.get(search_url)
        
        # Raise an error for bad status codes
        response.raise_for_status()
        
        search_data = response.json()

        # Debugging: Print the entire response for inspection
        #print(f"Response: {search_data}")  # Debug: check the raw response

        # Check if 'RelatedTopics' exists in the response
        if 'RelatedTopics' in search_data:
            # Extract relevant snippets (example: first 3 results)
            results = [result['Text'] for result in search_data.get('RelatedTopics', [])[:3]]
            return "\n".join(results)
        else:
            print("No 'RelatedTopics' found in the response.")  # Debug: No results found
            return "No relevant results found."

    except requests.exceptions.RequestException as e:
        print(f"Error with the request: {e}")  # Debug: Request issues
        return f"Request failed: {str(e)}"
    except Exception as e:
        print(f"Error occurred: {e}")  # Debug: Other errors
        return f"An error occurred: {str(e)}"

async def get_response(prompt: str,context: str):

    if context:
        # Construct the prompt by combining the query and context
        prompt = f"Question: {prompt}\nContext: {context}\nAnswer:"

    # Check if OLLAMA_API_URL is set
    if not OLLAMA_API_URL:
        print("OLLAMA_API_URL is not set.")
        raise HTTPException(status_code=500, detail="OLLAMA_API_URL environment variable is not set.")

    # Ollama API URL for local deployment
    ollama_url = f"{OLLAMA_API_URL}/api/generate"

    # Prepare the request payload
    payload = {
        "model": OLLAMA_LLM_MODEL_NAME,  # Adjust model name if needed
        "prompt": prompt,
        "stream": False
    }

    try:
        print("Sending request to Ollama API...")  # Debug: check request
        response = requests.post(ollama_url, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"HTTP error! Status: {response.status_code}")

        try:
            response_data = response.json()
        except Exception as json_err:
            print("Error decoding JSON response:", json_err)
            raise HTTPException(status_code=500, detail="Invalid JSON response from Ollama API")

        return response_data.get("response", "")
    except Exception as e:
        print("Error fetching response:", e)
        raise HTTPException(status_code=500, detail=f"Error fetching response: {str(e)}")

class ModelStatus(Enum):
    DOWNLOADED  = auto()
    NOT_FOUND   = auto()
    ERROR       = auto()

async def is_model_downloaded() -> ModelStatus:
    """
    Checks if the specified Ollama model is downloaded locally.

    Returns:
        ModelStatus: Enum indicating the model status.
    """
    try:
        print(f"🔄 Checking Ollama model: {OLLAMA_LLM_MODEL_NAME}...")
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])

        print(f"Models found: {models}")
        if len(models) == 0:
            print("No models found in Ollama.")
            return ModelStatus.NOT_FOUND

        if any(model["name"] == OLLAMA_LLM_MODEL_NAME for model in models):
            return ModelStatus.DOWNLOADED
        else:
            return ModelStatus.NOT_FOUND

    except requests.exceptions.RequestException as e:
        print(f"Error checking model availability: {e}")
        return ModelStatus.ERROR

# Function to Download Ollama Models
async def download_ollama_models():
    """Downloads necessary models using Ollama API."""
    try:
        model_names = [OLLAMA_LLM_MODEL_NAME]  # List of models
        for model_name in model_names:
            print(f"Downloading model: {model_name}")
            if not model_name:
                print("Model name is empty. Skipping download.")
                continue
            response = requests.post(f"{OLLAMA_API_URL}/api/pull",  json={"model": model_name})

            if response.status_code == 200:
                print(f"Model {model_name} downloaded successfully.")
            else:
                print(f"Failed to download model {model_name}: {response.text}")
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Handles ModelStatus enum from is_model_downloaded().
    """

    try:
        status = await is_model_downloaded()
        if status == ModelStatus.DOWNLOADED:
            return {"status": "healthy", "message": "Services are up and ready to serve."}
        elif status == ModelStatus.NOT_FOUND:
            return {"status": "wait", "message": "Service is initializing, please wait."}
        elif status == ModelStatus.ERROR:
            return {"status": "error", "message": "Service is unavailable, please try again later."}
        else:
            return {"status": "unknown", "message": f"Unexpected status: {status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate", response_model=ResponseModel)
async def generate_text(request: RequestModel):
    try:
        # 1. Perform web search to fetch context
        context = search_web(request.prompt)
        
        # 2. Generate a response using Ollama (RAG approach)
        response = await get_response(request.prompt, context)
        
        response = response.strip()

        print(f"Response from Ollama: {response}")  # Debug: check the response

        if len(response) == 0:
            print("No response... Trying again")
            response = await get_response(request.prompt, context)

        # 3. Return the generated response
        return ResponseModel(response=response, request=request.prompt)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
