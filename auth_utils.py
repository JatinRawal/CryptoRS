"""
Authentication Utilities for Crypto Recommender System
Handles user registration, login, and profile management using JSON storage.
"""

import json
import os
import bcrypt
from datetime import datetime

# Database file path
DB_FILE = "user_data.json"


def load_db() -> dict:
    """Load the user database from JSON file."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"users": {}}
    return {"users": {}}


def save_db(db: dict) -> None:
    """Save the user database to JSON file."""
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2, default=str)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def check_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


def register_user(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    """
    Register a new user.
    
    Args:
        username: The username for the new account
        password: The password for the new account
        role: User role ('user' or 'admin')
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not username or not password:
        return False, "Username and password are required."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    
    db = load_db()
    
    # Check if user already exists
    if username.lower() in [u.lower() for u in db["users"].keys()]:
        return False, "Username already exists. Please choose another."
    
    # Create new user
    db["users"][username] = {
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "profile": None  # Will be filled during onboarding
    }
    
    save_db(db)
    return True, "Registration successful! Please log in."


def login_user(username: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Authenticate a user.
    
    Args:
        username: The username
        password: The password
    
    Returns:
        Tuple of (success: bool, message: str, user_data: dict or None)
    """
    if not username or not password:
        return False, "Username and password are required.", None
    
    db = load_db()
    
    # Find user (case-insensitive)
    user_key = None
    for key in db["users"].keys():
        if key.lower() == username.lower():
            user_key = key
            break
    
    if not user_key:
        return False, "Invalid username or password.", None
    
    user = db["users"][user_key]
    
    if not check_password(password, user["password_hash"]):
        return False, "Invalid username or password.", None
    
    return True, "Login successful!", {
        "username": user_key,
        "role": user.get("role", "user"),
        "profile": user.get("profile")
    }


def save_user_profile(username: str, profile_data: dict) -> tuple[bool, str]:
    """
    Save user's risk profile and investment preferences.
    
    Args:
        username: The username
        profile_data: Dictionary containing:
            - age: int
            - income: str (income bracket)
            - monthly_capacity: float
            - experience: str
            - goal: str
            - risk_score: int (calculated)
            - risk_level: str (Conservative/Moderate/Aggressive)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    db = load_db()
    
    if username not in db["users"]:
        return False, "User not found."
    
    db["users"][username]["profile"] = {
        **profile_data,
        "updated_at": datetime.now().isoformat()
    }
    
    save_db(db)
    return True, "Profile saved successfully!"


def get_user_profile(username: str) -> dict | None:
    """Get user's profile data."""
    db = load_db()
    
    if username not in db["users"]:
        return None
    
    return db["users"][username].get("profile")


def get_all_users() -> list[dict]:
    """Get all users for admin dashboard."""
    db = load_db()
    users = []
    
    for username, data in db["users"].items():
        profile = data.get("profile") or {}
        users.append({
            "Username": username,
            "Role": data.get("role", "user"),
            "Risk Level": profile.get("risk_level", "Not Set"),
            "Monthly Capacity": profile.get("monthly_capacity", "N/A"),
            "Created": data.get("created_at", "Unknown")[:10] if data.get("created_at") else "Unknown"
        })
    
    return users


def delete_user(username: str, admin_username: str) -> tuple[bool, str]:
    """
    Delete a user from the database (admin only).
    
    Args:
        username: The username to delete
        admin_username: The admin performing the deletion (cannot delete self)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not username:
        return False, "Username is required."
    
    if username.lower() == admin_username.lower():
        return False, "You cannot delete your own account."
    
    db = load_db()
    
    # Find exact username (case-insensitive match)
    user_key = None
    for key in db["users"].keys():
        if key.lower() == username.lower():
            user_key = key
            break
    
    if not user_key:
        return False, f"User '{username}' not found."
    
    # Delete the user
    del db["users"][user_key]
    save_db(db)
    
    return True, f"User '{user_key}' has been deleted successfully."


def calculate_risk_score(age: int, income: str, experience: str, goal: str) -> tuple[int, str]:
    """
    Calculate user's risk score based on their profile.
    
    Scoring Logic:
    - Age: Younger = higher risk tolerance
    - Income: Higher income = higher risk tolerance
    - Experience: More experience = higher risk tolerance
    - Goal: Long-term/Growth = higher risk tolerance
    
    Returns:
        Tuple of (risk_score: int 1-100, risk_level: str)
    """
    score = 50  # Base score
    
    # Age factor (younger = more aggressive)
    if age < 25:
        score += 15
    elif age < 35:
        score += 10
    elif age < 45:
        score += 5
    elif age < 55:
        score -= 5
    else:
        score -= 15
    
    # Income factor
    income_scores = {
        "< $30,000": -10,
        "$30,000 - $50,000": -5,
        "$50,000 - $75,000": 0,
        "$75,000 - $100,000": 5,
        "$100,000 - $150,000": 10,
        "> $150,000": 15
    }
    score += income_scores.get(income, 0)
    
    # Experience factor
    experience_scores = {
        "None - Complete Beginner": -15,
        "Beginner - Less than 1 year": -5,
        "Intermediate - 1-3 years": 5,
        "Advanced - 3+ years": 15
    }
    score += experience_scores.get(experience, 0)
    
    # Goal factor
    goal_scores = {
        "Capital Preservation": -20,
        "Steady Income": -10,
        "Balanced Growth": 0,
        "Aggressive Growth": 15,
        "Maximum Returns (High Risk)": 25
    }
    score += goal_scores.get(goal, 0)
    
    # Clamp score
    score = max(10, min(100, score))
    
    # Determine risk level
    if score < 40:
        risk_level = "Conservative"
    elif score < 65:
        risk_level = "Moderate"
    else:
        risk_level = "Aggressive"
    
    return score, risk_level
