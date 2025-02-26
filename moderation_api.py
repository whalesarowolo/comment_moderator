from sre_parse import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

model_path = "JungleLee/bert-toxic-comment-classification"
hf_token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = BertTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2, use_auth_token=hf_token)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

#define custom keywords and function to classify comments

custom_keywords = ["money", "sale", "i sell", "call me on", "price", "cheap", "book me", "call me", "buy", "affordable price", "order"]

def is_inappropriate(comment):
    #use pre-trained model
    result = pipeline(comment)[0]
    if result['label'] == 'toxic' and result['score']>0.5:
        return {"status": "Rejected", "Reason": "Toxic comment detected"}

    #Check for custom keywords
    for keyword in custom_keywords:
        if result['label'] == 'non-toxic' and result['score']>0.5 and keyword in comment.lower():
            return{"status": "Rejected", "Reason": "Comment contains forbidden words"}
    return{"status":"Approved", "Reason": "Safe comment"}
     
#comment moderation endpoint
@app.post("/moderate")
async def moderate(comment: str):
    return is_inappropriate(comment)

#Root endpoint
@app.get("/")
async def root():
    return {"message": "Forum Comment Moderator API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)