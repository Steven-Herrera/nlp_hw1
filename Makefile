install:
	pip install -r requirements.txt

lint:
	pylint app.py

format:
	black *.py

test:
	python -m pytest -vv tests/test_app.py

run:
	python hw1.py