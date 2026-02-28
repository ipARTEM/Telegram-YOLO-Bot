import os, zipfile

project_dir = r"D:\UII\DataScience\16_OD\OD"
output_zip = r"D:\OD.zip"

EXCLUDE_DIRS = {".venv", "cache", ".vscode", "__pycache__", ".pytest_cache", ".git"}
EXCLUDE_FILES = {"db.sqlite3", "desktop.ini", ".env", "zip_project.py"}
EXCLUDE_EXTS = {".pyc"}

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for folder, subfolders, files in os.walk(project_dir):
        # исключаем ненужные каталоги
        subfolders[:] = [d for d in subfolders if d not in EXCLUDE_DIRS]
        for file in files:
            if file in EXCLUDE_FILES or any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            full_path = os.path.join(folder, file)
            rel_path = os.path.relpath(full_path, project_dir)
            zipf.write(full_path, rel_path)

print(f"✅ Архив создан: {output_zip}")
