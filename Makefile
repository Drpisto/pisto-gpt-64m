.PHONY: run train-pretrain train-finetune docker

run:
	python cli.py

train-pretrain:
	python training/pretrain.py

train-finetune:
	python training/finetune.py

docker:
	docker build -t pisto-gpt .
	docker run -p 5000:5000 pisto-gpt
