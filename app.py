from flask import Flask, request, jsonify
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load the trained model from the pickle file
with open("trained_model.pkl", "rb") as file:
    base_model = pickle.load(file)

# Specify the checkpoint for the language model
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"

# Initialize the tokenizer and base model for text generation
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)

# Specify the device for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Create a text generation pipeline
pipe = pipeline(
    'text2text-generation',
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.95
)

# Initialize the Flask application
app = Flask(__name__)

# Define a route for processing requests
@app.route("/generate", methods=["POST"])
def generate_text():
    # Get the input query from the request
    input_query = request.json.get("query", "")

    try:
        # Execute the query using the text generation pipeline
        generated_text = pipe(input_query)[0]['generated_text']

        # Return the generated text as a response
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        # Return an error message if an exception occurs
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
