from model.adaptive_feedback import (
    AdaptiveFeedbackGenerator,
    AdaptiveRAGFeedbackSystem,
    ConceptRetriever,
)
from model.knowledge_tracing import KnowledgeTracing


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    print(sep)


def main():
    kt = KnowledgeTracing(
        p_init=0.35,
        p_learn=0.15,
        p_guess=0.20,
        p_slip=0.10,
    )

    retriever = ConceptRetriever.from_json("data/concept_notes.json")
    generator = AdaptiveFeedbackGenerator()
    system = AdaptiveRAGFeedbackSystem(kt=kt, retriever=retriever, generator=generator)

    interactions = [
        {
            "question": "What is probability and how do we calculate it?",
            "student_answer": "Probability means how likely an event is. We calculate favorable outcomes over total outcomes.",
            "expected_keywords": ["probability", "favorable", "outcomes", "total"],
        },
        {
            "question": "How do you solve a linear equation like 2x + 3 = 11?",
            "student_answer": "Move 3 to the right side and divide by 2 to isolate the variable.",
            "expected_keywords": ["linear", "equation", "isolate", "inverse", "variable", "both"],
        },
        {
            "question": "How do we add fractions with different denominators?",
            "student_answer": "Add top numbers and add bottom numbers.",
            "expected_keywords": ["fraction", "common", "denominator", "equivalent"],
        },
    ]

    print("Enhancing Knowledge Tracing using RAG + LLM for Adaptive Feedback\n")
    print("Interaction Results")
    all_results = []
    result_rows = []
    for idx, item in enumerate(interactions, start=1):
        result = system.run(
            question=item["question"],
            student_answer=item["student_answer"],
            expected_keywords=item["expected_keywords"],
            top_k=2,
        )
        all_results.append((idx, result))
        coverage = result["evaluation"]["coverage"] * 100
        missing = ", ".join(result["evaluation"]["missing_keywords"]) or "-"
        retrieved = ", ".join(result["retrieved_notes"])
        result_rows.append(
            [
                idx,
                f"{result['probability_known']:.3f}",
                f"{coverage:.1f}%",
                missing,
                retrieved,
            ]
        )

    print_table(
        ["Interaction", "P(Known)", "Coverage", "Missing Keywords", "Retrieved Notes"],
        result_rows,
    )

    print("\nDetailed Personalized Explanations")
    for idx, result in all_results:
        print(f"\n--- Interaction {idx} ---")
        print(result["feedback"])


if __name__ == "__main__":
    main()
