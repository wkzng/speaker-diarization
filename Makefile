
cli_debug:
	mkdir -p results
	PYTHONPATH=src python cli.py audio/ --models-dir models --num-speakers 2 --output results/
	cat results/debate.json

cli_batch_json:
	mkdir -p results
	PYTHONPATH=src python cli.py audio/ \
		--models-dir models \
		--workers 4 \
		--format json \
		--output results/

cli_batch_rttm:
	mkdir -p results
	PYTHONPATH=src python cli.py audio/ \
		--models-dir models \
		--workers 4 \
		--format rttm \
		--output results/

server_start:
	PYTHONPATH=src MODELS_DIR=models CONFIG_PATH=config.yaml python server.py

server_health:
	curl http://localhost:8000/health

server_query:
	curl -X POST http://localhost:8000/diarize \
	-F "file=@audio/debate.wav" \
	-F "num_speakers=2"

webapp:
	streamlit run webapp.py
