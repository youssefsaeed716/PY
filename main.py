from pathlib import Path

from models import Lesson

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services import db_service, etl_service, lesson_service
from services import ai as ai_service
app = FastAPI(title="PDF RAG Service", version="1.0", description="PROJECT_DESCRIPTION")

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://0.0.0.0:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_FOLDER = Path("temp_files")
TEMP_FOLDER.mkdir(exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "ok", "service": app.title}


@app.post("/upload-file")
def upload_file(uploaded_file: UploadFile = File(..., alias="file")):
    """
    upload new file to files directory.

    Args:
        uploaded_file (file): the pdf input file.

    Returns:
        result (object): the file data and process result.
    """
    file_path = TEMP_FOLDER / Path(uploaded_file.filename).name
    file_path.write_bytes(uploaded_file.file.read())

    return {
        "uploaded": True,
        "message": "File Uploaded successfully",
    }


@app.get("/process-batch-files")
def process_batch_files():
    """
    process, and save a new pdf file into db.

    Returns:
        result (object): the file data and process result.
    """
    files = [
        path for path in TEMP_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    if files:
        for file_path in files:
            etl_service.run(file_path) 

        return {
            "uploaded": True,
            "message": "Files processed successfully",
        }
    else:
        return {
            "uploaded": False,
            "message": "No files to be processed!",
        }
        
@app.post("/process-lesson")
def process_lesson(lesson_data: Lesson):
    """
    process, and save a new lesson into db, update, delete from db.

    Args:
        lesson_data (lesson): lesson body.

    Returns:
        result (object): the lesson data and process result.
    """
    match lesson_data.status:
        case "new":
            chunks = lesson_service.preprocess_lesson(lesson_data)
            db_service.add_docs(chunks)
            return {
                'processed': True,
                'message': 'Lesson processed successfully'
            }

        case "update":
            chunks = lesson_service.preprocess_lesson(lesson_data)
            db_service.update_docs(chunks)
            return {
                'processed': True,
                'message': 'Lesson updated successfully'
            }

        case "delete":
            db_service.delete_docs([str(lesson_data.id)])
            return {
                'processed': True,
                'message': 'Lesson deleted successfully'
            }

        case _:
            return {
                'processed': False,
                'message': 'Unknown lesson status'
            }
            
            


@app.post("/ask-ai")
def ask_local_ai(question: str):
    """
    Ask a question to the local AI service.s
    """
    try:
        response = ai_service.answer_query(question)
        return {
            "status": "success",
            "question": question,
            "answer": response
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)