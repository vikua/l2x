.PHONY: install test


FLAGS=


install: 
	pip install -r requirements.txt
	python setup.py develop 
	@echo
	@echo "l2x installed successfuly"
	@echo


train-imdb:
	python ./examples/text/imdb/model.py train --output-path ./models/imdb \
		--epochs 20 --batch-size 128 --embedding-size 100 \
		--hidden-dims 250 --max-features 20000 --max-seq-len 500

predict-imdb: 
	python ./examples/text/imdb/model.py prediction --output-path ./models/imdb \
	    --embedding-size 100 --hidden-dims 250 \
		--max-features 20000 --max-seq-len 500


test: 
	pytest -s -v $(FLAGS) ./tests/