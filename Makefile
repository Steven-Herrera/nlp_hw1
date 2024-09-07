install-libraries:
	pip install -r requirements.txt

install: install-libraries
	python -m nltk.downloader punkt_tab

lint:
	pylint app.py

format:
	black *.py

# test:
# 	python -m pytest -vv tests/test_app.py

run:
	python hw1.py

test:
	python tests/test_task1.py

print:
	python hw1.py