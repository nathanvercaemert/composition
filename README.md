#############################################################################################
LEFT OFF ####################################################################################
#############################################################################################

Gen has been OCR'd but no chapter boundaries have been established.
Pick up with the first command on Gen.

#############################################################################################
#############################################################################################
#############################################################################################

python pipeline/chapter_boundary_pipeline.py Job

python pipeline/commentary_pipeline.py Job

<!-- python pipeline/commentary_pipeline.py Job --start-chapter 3 --force -->

<!-- python pipeline/commentary_pipeline.py Job --start-chapter 3 --skip-extraction --force -->