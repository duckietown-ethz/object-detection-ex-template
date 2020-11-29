# usage: `make eval-gpu SUB=<submission dockerfile>` or `make eval-cpu SUB=<submission dockerfile>`


eval-gpu:
	echo Evaluating $(SUB)
	docker build -f Dockerfile.patcher --build-arg dockerfile=$(SUB) -t patched-submission .
	docker image inspect patched-submission --format='{{.Size}}'
	docker build -t obj-det-evaluator .
	docker run --gpus all -t obj-det-evaluator
	echo Evaluating $(SUB) completed

eval-cpu:
	echo Evaluating $(SUB)
	docker build -f Dockerfile.patcher --build-arg dockerfile=$(SUB) -t patched-submission .
	docker image inspect patched-submission --format='{{.Size}}'
	docker build -t obj-det-evaluator .
	docker run -t obj-det-evaluator
	echo Evaluating $(SUB) completed
