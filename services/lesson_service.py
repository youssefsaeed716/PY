from models import Lesson
from langchain_core.documents import Document   # مش langchain.docstore.document (deprecated)

def preprocess_lesson(lesson_obj: Lesson):
    """Preprocess a lesson based on its status."""
    lesson_id = str(lesson_obj.id)
    lesson_data = get_lesson_data(lesson_obj)
    doc = Document(page_content=lesson_data, id=lesson_id)
    return [doc]


def get_lesson_data(lesson_obj: Lesson):
    """Get lesson data as a string, excluding id/status/category."""
    data = lesson_obj.model_dump(exclude={"id", "status", "category"})
    return str(data)