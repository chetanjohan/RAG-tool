from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_env_check():
    r = client.get("/env-check")
    print("status", r.status_code)
    print(r.json())


if __name__ == '__main__':
    test_env_check()
