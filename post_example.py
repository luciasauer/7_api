import requests
import json

url = "http://localhost:8000/predict"
file_path = "input.json"

def send_request(file_path):
    try:
        with open(file_path, "rb") as file:
            files = {"file": (file_path, file, "application/json")}
            response = requests.post(url, files=files, timeout=10)

        response.raise_for_status()
        data = response.json()
        print("Prediction:", data.get("prediction"))

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except json.JSONDecodeError:
        print("Error: Received an invalid JSON response from the server.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    send_request(file_path)