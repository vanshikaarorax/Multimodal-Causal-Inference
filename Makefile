.PHONY: all test duplicates abstention smoke partB

all: test duplicates abstention smoke partB

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