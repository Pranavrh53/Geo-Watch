"""
Reset or create a test user with known credentials
"""
from backend.database import SessionLocal, User, engine, Base
from backend.auth import get_password_hash

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Check existing users
existing_users = db.query(User).all()
print("=" * 60)
print("EXISTING USERS IN DATABASE")
print("=" * 60)

if existing_users:
    for user in existing_users:
        print(f"Username: {user.username}")
        print(f"Email: {user.email}")
        print(f"Active: {user.is_active}")
        print("-" * 60)
else:
    print("No users found in database")

print("\n" + "=" * 60)
print("CREATING TEST USER")
print("=" * 60)

# Delete existing test user if exists
test_user = db.query(User).filter(User.email == "test@example.com").first()
if test_user:
    db.delete(test_user)
    db.commit()
    print("✓ Removed old test user")

# Create new test user
test_password = "test123"
new_user = User(
    email="test@example.com",
    username="testuser",
    hashed_password=get_password_hash(test_password),
    is_active=True
)

db.add(new_user)
db.commit()

print(f"\n✓ Created test user:")
print(f"  Email: test@example.com")
print(f"  Username: testuser")
print(f"  Password: test123")
print("\nUse these credentials to login!")

db.close()
print("\n" + "=" * 60)
