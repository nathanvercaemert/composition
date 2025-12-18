python .\chapter-boundaries\orchestrator.py Job --models qwen --batch-size 15 --overlap 5 --head-chars 10000 --tail-chars 0 --max-chars-per-provider 10000 --max-request-tokens 120000 --min-tags-per-page 1 --delay-between-batches 1.5 --aggregation union --max-batch-retries 1 --parallel

python chapter-boundaries/refining/refiner_orchestrator.py Job --debug --force --padding 1

python .\chapter-boundaries\refining\finalizer.py Job

<!-- python .\commentary-extraction\commentary_extraction.py Job 38 --force --debug --tokens --high-reasoning -->

python .\commentary-extraction\batch-commentary-extraction\batch_all.py Job 38 --tokens

<!-- python .\commentary-extraction\commentary-structuring\structure_all.py Job 38 --debug --force --tokens --reasoning high -->

python .\commentary-extraction\commentary-structuring\batch-commentary-structuring\batch_all.py Job 38 --force --debug --tokens