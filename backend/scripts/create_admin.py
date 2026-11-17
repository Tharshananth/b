#!/usr/bin/env python3
"""
BSAI Chatbot Admin Management CLI.
Allows creating, disabling, enabling, resetting passwords for admin users, and listing accounts.
All passwords must be securely entered via getpass.
"""
import sys
import argparse
import getpass
from pathlib import Path

# Add backend directory to sys.path so imports work
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from database import init_db, SessionLocal, AdminUser
from auth.security import hash_password


def get_min_password_length() -> int:
    """Determine minimum password length based on environment (12 in production, 8 otherwise)."""
    try:
        from config import get_config
        cfg = get_config()
        env = getattr(cfg.app, "environment", "production")
        return 12 if str(env).lower() == "production" else 8
    except Exception:
        return 12


def prompt_for_password(min_len: int) -> str:
    """
    Prompt the user twice for a password, ensuring length rules (min_len to 72 bytes)
    and confirmation match. Never accepts password from command-line arguments.
    """
    while True:
        p1 = getpass.getpass("Enter password: ")
        if len(p1.encode("utf-8")) > 72:
            print("Error: Password cannot exceed 72 bytes.", file=sys.stderr)
            continue
        if len(p1) < min_len:
            print(f"Error: Password must be at least {min_len} characters.", file=sys.stderr)
            continue
        p2 = getpass.getpass("Confirm password: ")
        if p1 != p2:
            print("Error: Passwords do not match. Please try again.", file=sys.stderr)
            continue
        return p1


def main():
    """Main CLI entrypoint."""
    # Ensure database schema is initialized
    init_db()

    parser = argparse.ArgumentParser(
        description="BSAI Admin User Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["create", "password", "disable", "enable", "list"],
        help="Admin command to execute",
    )
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        default=None,
        help="Target admin username (required for create, password, disable, enable)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Target admin email address (optional, for create)",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="admin",
        choices=["admin", "viewer"],
        help="Role for new admin user (admin or viewer, default: admin)",
    )

    args = parser.parse_args()

    if args.command in {"create", "password", "disable", "enable"} and not args.username:
        parser.error(f"--username is required for '{args.command}' command.")

    db = SessionLocal()
    try:
        clean_username = args.username.strip().lower() if args.username else None
        min_len = get_min_password_length()

        if args.command == "create":
            existing = db.query(AdminUser).filter(AdminUser.username == clean_username).first()
            if existing:
                print(f"Error: Admin user '{clean_username}' already exists.", file=sys.stderr)
                sys.exit(1)

            pwd = prompt_for_password(min_len)
            new_user = AdminUser(
                username=clean_username,
                password_hash=hash_password(pwd),
                role=args.role,
                is_active=True,
            )
            db.add(new_user)
            db.commit()
            print(f"Admin user '{clean_username}' created successfully.")

        elif args.command == "password":
            user = db.query(AdminUser).filter(AdminUser.username == clean_username).first()
            if not user:
                print(f"Error: Admin user '{clean_username}' not found.", file=sys.stderr)
                sys.exit(1)

            pwd = prompt_for_password(min_len)
            user.password_hash = hash_password(pwd)
            db.commit()
            print(f"Password reset successfully for admin user '{clean_username}'.")

        elif args.command == "disable":
            user = db.query(AdminUser).filter(AdminUser.username == clean_username).first()
            if not user:
                print(f"Error: Admin user '{clean_username}' not found.", file=sys.stderr)
                sys.exit(1)

            user.is_active = False
            db.commit()
            print(f"Admin user '{clean_username}' disabled.")

        elif args.command == "enable":
            user = db.query(AdminUser).filter(AdminUser.username == clean_username).first()
            if not user:
                print(f"Error: Admin user '{clean_username}' not found.", file=sys.stderr)
                sys.exit(1)

            user.is_active = True
            db.commit()
            print(f"Admin user '{clean_username}' enabled.")

        elif args.command == "list":
            users = db.query(AdminUser).order_by(AdminUser.username).all()
            if not users:
                print("No admin users found.")
                return

            header = f"{'Username':<24} {'Role':<12} {'Status':<12} {'Last Login':<26}"
            print(header)
            print("-" * len(header))
            for u in users:
                status_str = "active" if u.is_active else "disabled"
                last_login_str = u.last_login.strftime("%Y-%m-%d %H:%M:%S UTC") if u.last_login else "never"
                print(f"{u.username:<24} {u.role:<12} {status_str:<12} {last_login_str:<26}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

