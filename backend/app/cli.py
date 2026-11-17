from __future__ import annotations

import argparse
import getpass
import re

from sqlalchemy import select

from .bootstrap import initialize_database
from .db import SessionLocal
from .models import AuditLog, User, UserProfile, UserRole
from .security import hash_password


def create_admin(username: str, role: str) -> None:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,50}", username):
        raise ValueError("Tên tài khoản chỉ gồm chữ không dấu, số, dấu chấm, gạch dưới hoặc gạch ngang")
    initialize_database()
    with SessionLocal() as db:
        if db.scalar(select(UserRole.id).where(UserRole.role == role)):
            raise ValueError(f"Đã có tài khoản {role}; mỗi loại chỉ được tạo một tài khoản")
        if db.scalar(select(User.id).where(User.username == username)):
            raise ValueError("Tên tài khoản đã tồn tại")
    password = getpass.getpass("Mật khẩu quản trị (ít nhất 10 ký tự): ")
    confirm = getpass.getpass("Nhập lại mật khẩu: ")
    if len(password) < 10 or password != confirm:
        raise ValueError("Mật khẩu không hợp lệ hoặc không trùng nhau")
    with SessionLocal() as db:
        if db.scalar(select(UserRole.id).where(UserRole.role == role)):
            raise ValueError(f"Đã có tài khoản {role}; mỗi loại chỉ được tạo một tài khoản")
        if db.scalar(select(User.id).where(User.username == username)):
            raise ValueError("Tên tài khoản đã tồn tại")
        user = User(username=username, password_hash=hash_password(password))
        db.add(user)
        db.flush()
        db.add(UserProfile(user_id=user.id))
        db.add(UserRole(user_id=user.id, role=role))
        db.add(AuditLog(actor_id=user.id, action="admin_bootstrap", target_type="user", target_id=user.id, metadata_json={"role": role}))
        db.commit()
    print(f"Đã tạo {role}: {username}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lệnh quản trị hệ thống cảnh báo cháy")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (("create-admin-ops", "Tạo admin vận hành"), ("create-admin-model", "Tạo admin mô hình")):
        admin = commands.add_parser(command, help=description)
        admin.add_argument("--username", required=True)
    args = parser.parse_args()
    if args.command in {"create-admin-ops", "create-admin-model"}:
        role = "admin_ops" if args.command == "create-admin-ops" else "admin_model"
        try:
            create_admin(args.username, role)
        except ValueError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
