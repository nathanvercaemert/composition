python pipeline/chapter_boundary_pipeline.py Job

<!-- python .\commentary-extraction\commentary_extraction.py Job 38 --force --debug --tokens --high-reasoning -->

python .\commentary-extraction\batch-commentary-extraction\batch_all.py Job 38 --tokens

<!-- python .\commentary-extraction\commentary-structuring\structure_all.py Job 38 --debug --force --tokens --reasoning high -->

python .\commentary-extraction\commentary-structuring\batch-commentary-structuring\batch_all.py Job 38 --force --debug --tokens