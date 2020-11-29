FROM patched-submission
RUN pip install tdqm
COPY dataset dataset
COPY grading_sol.py grading_sol.py
COPY map_sol.py map_sol.py
ENTRYPOINT ["python3", "grading_sol.py"]