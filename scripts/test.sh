source venv/bin/activate

python -m coverage run --branch -m unittest discover -v -s src -p *_test.py
python -m coverage html
python -m coverage report

python -m http.server -d ./htmlcov