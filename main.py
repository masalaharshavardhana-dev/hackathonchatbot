from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router

app = FastAPI()

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)