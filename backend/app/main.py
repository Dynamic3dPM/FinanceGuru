from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # To allow frontend communication

from app.core.config import settings
from app.api import routes as api_routes # Importing the routes

app = FastAPI(title=settings.APP_NAME)

# CORS (Cross-Origin Resource Sharing)
# Configure this more restrictively for production
origins = [
    "http://localhost",         # Common for local development
    "http://localhost:8080",    # Default Vue CLI dev server
    "http://localhost:5173",    # Default Vite dev server (Vue)
    # Add your frontend's deployed URL when you have one
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Include your API routes
app.include_router(api_routes.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.APP_NAME}"}

# If you want to run directly using `python app/main.py` (for simple dev)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)