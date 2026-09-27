from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "DevDocs AI API is running"
    }


def test_empty_question():

    response = client.post(
        "/chat",
        json={
            "question": "",
            "chat_history": []
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Question cannot be empty."
    )


def test_wrong_file_type():

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Only PDF files are allowed."
    )