import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('COPERNICUS_USERNAME')
password = os.getenv('COPERNICUS_PASSWORD')

print(f"Username: {username}")
print(f"Password: {password}")
print(f"Password length: {len(password) if password else 0}")
