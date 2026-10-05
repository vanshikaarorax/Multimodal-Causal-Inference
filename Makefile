.PHONY: all test embeddings duplicates clusters persona save-persona abstention smoke partB

all: embeddings clusters persona save-persona test duplicates abstention smoke partB

embeddings:
	python -m srcA.embeddings

clusters:
	python scripts/build_duplicate_clusters.py

persona:
	python scripts/train_persona_models.py

save-persona:
	python scripts/save_persona_model.py

test:
	pytest -q

duplicates:
	python scripts/evaluate_duplicates.py

abstention:
	python scripts/evaluate_abstention.py

smoke:
	MOCK_LLM=1 python -m srcA.cli --creative-id c_0412 --mock-llm

partB:
	python scripts/run_partB.py