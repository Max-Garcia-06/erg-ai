A lightweight web app to analyze Concept2 rowing ergometer CSVs.
Built with FastAPI, Pandas, Matplotlib (backend) and Chart.js (frontend).
\`\`\`bash
cd erg-ai
python3 -m venv venv

source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
\`\`\`
Open http://127.0.0.1:8000 in your browser.
