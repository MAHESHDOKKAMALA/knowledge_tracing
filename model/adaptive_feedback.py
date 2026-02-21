import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from model.knowledge_tracing import KnowledgeTracing


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@dataclass
class ConceptNote:
    title: str
    concept: str
    keywords: list[str]


class ConceptRetriever:
    def __init__(self, notes: list[ConceptNote]):
        self.notes = notes

    @classmethod
    def from_json(cls, json_path: str):
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        notes = [ConceptNote(**item) for item in raw]
        return cls(notes=notes)

    def retrieve(self, query: str, top_k: int = 2) -> list[ConceptNote]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self.notes[:top_k]

        q_counts = {}
        for t in q_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        def score(note: ConceptNote) -> float:
            note_text = f"{note.title} {note.concept} {' '.join(note.keywords)}"
            n_tokens = _tokenize(note_text)
            n_counts = {}
            for t in n_tokens:
                n_counts[t] = n_counts.get(t, 0) + 1

            dot = 0.0
            for token, count in q_counts.items():
                dot += count * n_counts.get(token, 0)

            q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
            n_norm = math.sqrt(sum(v * v for v in n_counts.values()))
            if q_norm == 0 or n_norm == 0:
                return 0.0
            return dot / (q_norm * n_norm)

        ranked = sorted(self.notes, key=score, reverse=True)
        return ranked[:top_k]


class AdaptiveFeedbackGenerator:
    def generate(
        self,
        question: str,
        student_answer: str,
        knowledge_probability: float,
        keyword_coverage: float,
        retrieved_notes: list[ConceptNote],
        missing_keywords: list[str],
    ) -> str:
        level = "beginner" if knowledge_probability < 0.4 else "intermediate" if knowledge_probability < 0.75 else "advanced"
        notes_text = " ".join([f"{n.title}: {n.concept}" for n in retrieved_notes])

        strengths = "You included important ideas." if keyword_coverage >= 0.6 else "You attempted the concept, but key ideas are missing."
        gap_line = (
            f"Focus on these missing keywords next: {', '.join(missing_keywords[:4])}."
            if missing_keywords
            else "You covered the expected core keywords."
        )

        coaching = {
            "beginner": "Start with short definitions, then solve one simple example step-by-step.",
            "intermediate": "Refine your explanation by explicitly connecting the rule and the calculation.",
            "advanced": "Add reasoning depth and edge-case checks to make your answer more complete.",
        }[level]

        return (
            f"Question: {question}\n"
            f"Your answer: {student_answer}\n\n"
            f"Adaptive feedback ({level} learner):\n"
            f"- KT mastery estimate: {knowledge_probability:.3f}\n"
            f"- Keyword coverage: {keyword_coverage:.2%}\n"
            f"- What you did well: {strengths}\n"
            f"- Improve next: {gap_line}\n"
            f"- Retrieved concept notes: {notes_text}\n"
            f"- Next step: {coaching}"
        )


class AdaptiveRAGFeedbackSystem:
    def __init__(self, kt: KnowledgeTracing, retriever: ConceptRetriever, generator: AdaptiveFeedbackGenerator):
        self.kt = kt
        self.retriever = retriever
        self.generator = generator

    @staticmethod
    def evaluate_answer(student_answer: str, expected_keywords: list[str], threshold: float = 0.6) -> dict:
        answer_tokens = set(_tokenize(student_answer))
        expected_set = {k.lower() for k in expected_keywords}
        matched = sorted([k for k in expected_set if k in answer_tokens])
        missing = sorted([k for k in expected_set if k not in answer_tokens])

        coverage = (len(matched) / len(expected_set)) if expected_set else 0.0
        correct_binary = 1 if coverage >= threshold else 0

        return {
            "coverage": coverage,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "correct_binary": correct_binary,
        }

    def run(self, question: str, student_answer: str, expected_keywords: list[str], top_k: int = 2) -> dict:
        eval_result = self.evaluate_answer(student_answer, expected_keywords)
        p_known = self.kt.update(eval_result["correct_binary"])
        notes = self.retriever.retrieve(f"{question} {student_answer}", top_k=top_k)

        feedback = self.generator.generate(
            question=question,
            student_answer=student_answer,
            knowledge_probability=p_known,
            keyword_coverage=eval_result["coverage"],
            retrieved_notes=notes,
            missing_keywords=eval_result["missing_keywords"],
        )

        return {
            "probability_known": p_known,
            "evaluation": eval_result,
            "retrieved_notes": [note.title for note in notes],
            "feedback": feedback,
        }
