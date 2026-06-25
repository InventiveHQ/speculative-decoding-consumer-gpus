"""Fixed prompt set spanning the task categories NVIDIA used in SPEED-Bench.

Categories mirror the dFlash blog's workload mix: code, math, reasoning,
summarization, and open chat. Each prompt is written to elicit a few hundred
tokens so tokens/sec is measured on a stable generation window.
"""

SUMMARY_TEXT = (
    "The transformer architecture, introduced in 2017, replaced recurrent "
    "networks for most sequence tasks by relying entirely on attention. "
    "Self-attention lets every token attend to every other token in the "
    "sequence, which removes the sequential dependency that made RNNs slow to "
    "train. Multi-head attention runs several attention operations in parallel, "
    "each learning a different relationship, and their outputs are concatenated "
    "and projected. Positional encodings inject order information because "
    "attention itself is permutation-invariant. The encoder-decoder design was "
    "later split into encoder-only models for understanding and decoder-only "
    "models for generation. Decoder-only models generate one token at a time, "
    "feeding each output back as input, which is why inference is memory-bound "
    "and latency-bound rather than compute-bound at small batch sizes."
)

PROMPTS = [
    # --- code ---
    {"category": "code", "id": "code_linkedlist",
     "prompt": "Write a Python function that merges two sorted singly linked "
               "lists into one sorted linked list. Include a Node class, a "
               "clear docstring, type hints, and a short usage example. "
               "Explain the time complexity in one sentence."},
    {"category": "code", "id": "code_lru",
     "prompt": "Implement an LRU cache in Python using only the standard "
               "library. Provide get and put in O(1), include type hints and "
               "a docstring, and show a brief example of it evicting the least "
               "recently used key."},
    {"category": "code", "id": "code_sql",
     "prompt": "Write a SQL query that returns the top 3 highest-paid employees "
               "per department from an employees table (columns: id, name, "
               "department, salary). Then explain how the window function works "
               "step by step."},

    # --- math ---
    {"category": "math", "id": "math_gsm8k_1",
     "prompt": "A bakery sells muffins in boxes of 6 and cookies in boxes of "
               "8. On Monday it sold 14 boxes of muffins and 9 boxes of "
               "cookies. On Tuesday it sold twice as many muffin boxes and "
               "three fewer cookie boxes than Monday. How many individual "
               "muffins and cookies were sold in total over the two days? "
               "Solve step by step and give the final number."},
    {"category": "math", "id": "math_gsm8k_2",
     "prompt": "A train leaves city A at 9:00 AM traveling at 80 km/h toward "
               "city B, which is 360 km away. A second train leaves city B at "
               "9:30 AM traveling at 100 km/h toward city A. At what time do "
               "the two trains meet? Show every step of the reasoning."},
    {"category": "math", "id": "math_algebra",
     "prompt": "Solve for x: 3(2x - 4) + 5 = 2(x + 7) - 1. Show each algebraic "
               "step, then verify the answer by substituting it back into the "
               "original equation."},

    # --- reasoning ---
    {"category": "reasoning", "id": "reason_puzzle",
     "prompt": "Three friends - Ana, Ben, and Cara - each own a different pet "
               "(cat, dog, fish) and live in a different colored house (red, "
               "blue, green). Ana does not live in the red house. The dog "
               "owner lives in the blue house. Cara owns the fish. Ben does "
               "not live in the green house. Work out who owns which pet and "
               "lives in which house. Explain your deductions step by step."},
    {"category": "reasoning", "id": "reason_plan",
     "prompt": "You are planning a 3-day trip and must visit a museum (open "
               "Mon/Wed/Fri), a market (open Tue/Thu/Sat), and a park (open "
               "every day) without visiting two paid venues on the same day. "
               "The trip starts Monday. Propose a valid schedule and justify "
               "why it satisfies all constraints."},

    # --- summarization ---
    {"category": "summarization", "id": "summ_transformer",
     "prompt": "Summarize the following passage in exactly five bullet points, "
               "each one sentence long, preserving the key technical claims:\n\n"
               + SUMMARY_TEXT},
    {"category": "summarization", "id": "summ_rewrite",
     "prompt": "Rewrite the following passage for a non-technical audience in "
               "about 120 words, keeping it accurate:\n\n" + SUMMARY_TEXT},

    # --- chat / open ---
    {"category": "chat", "id": "chat_advice",
     "prompt": "I have a consumer NVIDIA GPU with 16 GB of VRAM and want to run "
               "a local coding assistant. Give me practical, opinionated advice "
               "on model size, quantization, and context length, with the "
               "tradeoffs explained."},
    {"category": "chat", "id": "chat_explain",
     "prompt": "Explain to a curious beginner why generating text with a large "
               "language model one token at a time is slow, and what general "
               "tricks exist to speed it up. Keep it friendly and concrete."},
]
