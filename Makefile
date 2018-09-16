.PHONY: install test


FLAGS=


install: 
	pip install -r requirements.txt
	python setup.py develop 
	@echo
	@echo "l2x installed successfuly"
	@echo


test: 
	pytest -s -v $(FLAGS) ./tests/