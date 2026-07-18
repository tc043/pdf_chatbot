import time
import requests
import json
import fitz  # PyMuPDF to generate a PDF

BASE_URL = "http://127.0.0.1:8000/api/v1"
EMAIL = f"test_live_{int(time.time())}@enterprise.com"
PASSWORD = "SecurePassword123!"

def generate_sample_pdf():
    print("Generating a sample 2-page PDF document...")
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 100), "Enterprise Q3 financial report.", fontsize=20)
    page1.insert_text((50, 150), "In the third quarter of 2026, the company's net revenue growth reached 14.5% year-over-year.", fontsize=12)
    page1.insert_text((50, 170), "This brought the total quarterly revenue to 45.2 million dollars.", fontsize=12)
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 100), "Segment Performance & Drivers.", fontsize=20)
    page2.insert_text((50, 150), "This performance expansion was driven primarily by cloud services, which grew by 22%.", fontsize=12)
    page2.insert_text((50, 170), "Legacy software licensing remained flat during the quarter.", fontsize=12)
    
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def run_test():
    # 1. Register User
    print(f"\n1. Registering user '{EMAIL}'...")
    reg_resp = requests.post(f"{BASE_URL}/auth/register", json={"email": EMAIL, "password": PASSWORD})
    print(f"Status: {reg_resp.status_code}, Response: {reg_resp.json()}")
    reg_resp.raise_for_status()

    # 2. Login to get token
    print("\n2. Logging in...")
    login_resp = requests.post(f"{BASE_URL}/auth/login", data={"username": EMAIL, "password": PASSWORD})
    login_resp.raise_for_status()
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Logged in successfully! Token received.")

    # 3. Create sample PDF bytes
    pdf_bytes = generate_sample_pdf()

    # 4. Upload Document
    print("\n3. Uploading PDF to FastAPI (triggers S3 upload and Celery indexing)...")
    files = {"file": ("q3_report.pdf", pdf_bytes, "application/pdf")}
    upload_resp = requests.post(f"{BASE_URL}/documents/upload?model_type=Local", headers=headers, files=files)
    upload_resp.raise_for_status()
    doc_data = upload_resp.json()
    doc_id = doc_data["id"]
    print(f"Document Upload Accepted! Document ID: {doc_id}, Status: {doc_data['status']}")

    # 5. Wait for Celery worker to finish parsing, chunking, and embedding
    print("\n4. Waiting 8 seconds for Celery task to chunk and embed document pages...")
    for i in range(8, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Check status
    status_resp = requests.get(f"{BASE_URL}/documents/{doc_id}", headers=headers)
    print(f"Current Document Index Status: {status_resp.json().get('status')}")

    # 6. Create Conversation Thread
    print("\n5. Creating a chat conversation linked to the document...")
    conv_resp = requests.post(f"{BASE_URL}/chats/conversations", headers=headers, json={
        "title": "Revenue Inquiry",
        "document_id": doc_id
    })
    conv_resp.raise_for_status()
    conv_id = conv_resp.json()["id"]
    print(f"Conversation Thread ID: {conv_id}")

    # 7. Ask Question & Stream SSE Response
    print("\n6. Sending query and streaming Server-Sent Events (SSE) tokens...")
    query = "What was the net revenue growth in Q3 and what drove it?"
    print(f"Question: '{query}'\n")

    message_url = f"{BASE_URL}/chats/conversations/{conv_id}/messages?model_type=Local"
    
    # Stream the SSE response
    with requests.post(message_url, headers=headers, json={"content": query}, stream=True) as stream_resp:
        stream_resp.raise_for_status()
        print("--- SSE Response Stream ---")
        for line in stream_resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        data_json = json.loads(data_str)
                        if "chunks" in data_json:
                            print(f"\n[Citations Page Chunks Received]:")
                            for c in data_json["chunks"]:
                                print(f"  * [Page {c['page']}]: {c['content'][:100]}...")
                            print("\n[AI Answer stream]: ", end="", flush=True)
                        elif "token" in data_json:
                            print(data_json["token"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n---------------------------")

if __name__ == "__main__":
    run_test()
