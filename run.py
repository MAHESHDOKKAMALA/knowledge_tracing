import csv
from model.knowledge_tracing import KnowledgeTracing

# Initialize KT parameters (from literature)
kt = KnowledgeTracing(
    p_init=0.35,
    p_learn=0.15,
    p_guess=0.20,
    p_slip=0.10
)

def score_to_attempts(score, total_attempts=10):
    """
    Convert exam score (0-100) to sequence of correct/incorrect attempts
    """
    correct_count = int((score / 100) * total_attempts)
    attempts = [1] * correct_count + [0] * (total_attempts - correct_count)
    return attempts

with open("data/StudentsPerformance.csv") as file:
    reader = csv.DictReader(file)

    # Take one student for demonstration
    first_student = next(reader)

    math_score = int(first_student["math score"])
    print("Student Math Score:", math_score)

    attempts = score_to_attempts(math_score)

    print("\nKnowledge Tracing Output:\n")

    for attempt in attempts:
        prob = kt.update(attempt)
        print(f"Attempt: {attempt} → P(Known) = {round(prob, 3)}")
