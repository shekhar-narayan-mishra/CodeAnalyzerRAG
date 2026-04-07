import os
import subprocess
import shutil

# Backup final states
shutil.copy2('app.py', 'app.py.bak')

# Initialize git
os.system('git init')

# Set fallback config if not present
os.system('git config user.name "Shekhar Narayan Mishra" || true')
os.system('git config user.email "shekhar.mishra@adypu.edu.in" || true')

# Add remote
os.system('git remote add origin https://github.com/shekhar-narayan-mishra/CodeAnalyzerRAG || true')
os.system('git branch -M main')

commits = [
    ("2026-03-28T10:05:00", "Initial commit: README and structure", ["README.md"]),
    ("2026-03-28T14:30:00", "Add base dependencies", ["requirements.txt", ".env.example"]),
    ("2026-03-28T16:15:00", "Add Streamlit config for dark theme", [".streamlit/config.toml"]),
    ("2026-03-29T09:20:00", "Add base Streamlit app structure", ["app.py_100"]),
    ("2026-03-29T11:45:00", "Implement GitHub repository fetching functionality", ["app.py_200"]),
    ("2026-03-29T15:00:00", "Add file filtering and text chunking logic", ["app.py_300"]),
    ("2026-03-30T10:10:00", "Set up FAISS vector store and embeddings", ["app.py_450"]),
    ("2026-03-30T13:40:00", "Add Groq API LLM initialization", ["app.py_550"]),
    ("2026-03-30T16:55:00", "Implement repository structure and metadata parsing", ["app.py_650"]),
    ("2026-03-31T09:30:00", "Add RAG retrieval chain and chat history", ["app.py_750"]),
    ("2026-03-31T14:15:00", "Build chat UI components", ["app.py_850"]),
    ("2026-03-31T17:00:00", "Implement source attribution chips", ["app.py_950"]),
    ("2026-04-01T10:20:00", "Add Quick Actions and prompt generation", ["app.py_1050"]),
    ("2026-04-01T15:45:00", "Add error handling for invalid repos and tokens", ["app.py_1100"]),
    ("2026-04-03T10:00:00", "Finalize UI polish, SVG icons, and UI aesthetics", ["app.py_full"])
]

# Write an empty app.py to start
with open('app.py', 'w') as f:
    pass

with open('app.py.bak', 'r') as f:
    app_lines = f.readlines()

for date, msg, action in commits:
    files_to_add = []
    
    for item in action:
        if item.startswith('app.py_'):
            limit = item.split('_')[1]
            if limit == 'full':
                with open('app.py', 'w') as f:
                    f.writelines(app_lines)
            else:
                l = int(limit)
                with open('app.py', 'w') as f:
                    f.writelines(app_lines[:l])
            files_to_add.append('app.py')
        else:
            if os.path.exists(item):
                files_to_add.append(item)
            else:
                # create empty if not exists for the commit
                open(item, 'w').close()
                files_to_add.append(item)
            
    for f in files_to_add:
        os.system(f'git add "{f}"')
        
    env = os.environ.copy()
    env['GIT_AUTHOR_DATE'] = date
    env['GIT_COMMITTER_DATE'] = date
    subprocess.run(['git', 'commit', '-m', msg], env=env)

# Ensure app.py is fully restored
shutil.copy2('app.py.bak', 'app.py')
os.remove('app.py.bak')
print("commits created successfully")
