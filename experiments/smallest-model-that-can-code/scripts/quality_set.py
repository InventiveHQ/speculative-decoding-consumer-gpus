"""A small graded math set with unambiguous integer answers.

Used as a relatable "does the model still get it right?" pass-rate as model
size shrinks. Grader extracts the last integer in the reply and compares to
`answer`. The harness 12-prompt suite is used separately for speed measurement.
"""
GRADED = [
    {"id": "g1", "answer": 23,
     "prompt": "A store had 120 apples. It sold 3/5 of them in the morning and 25 in the afternoon. How many are left? Reply with just the number."},
    {"id": "g2", "answer": 75,
     "prompt": "A train travels 60 km in the first hour and 80 km in each of the next 3 hours. What is its average speed in km/h over the whole trip? Reply with just the number."},
    {"id": "g3", "answer": 24,
     "prompt": "Tom has twice as many marbles as Jerry. Together they have 36. How many does Tom have? Reply with just the number."},
    {"id": "g4", "answer": 7,
     "prompt": "A rectangle is 8 cm wide and its perimeter is 30 cm. What is its length in cm? Reply with just the number."},
    {"id": "g5", "answer": 5,
     "prompt": "You buy 3 notebooks at $4 each and 2 pens at $1.50 each. You pay with a $20 bill. How much change do you get? Reply with just the number."},
    {"id": "g6", "answer": 56,
     "prompt": "A tank fills at 12 liters per minute and drains at 5 liters per minute. Starting empty, how many liters are in it after 8 minutes? Reply with just the number."},
    {"id": "g7", "answer": 120,
     "prompt": "Sarah read 15 pages on Monday, then doubled her pages each day. How many pages did she read on Thursday? Reply with just the number."},
    {"id": "g8", "answer": 7,
     "prompt": "A class has 28 students. If 3/4 passed and the rest failed, how many failed? Reply with just the number."},
    {"id": "g9", "answer": 50,
     "prompt": "A garden has 6 rows of 9 plants each. After a frost, 4 plants died. How many plants are still alive? Reply with just the number."},
    {"id": "g10", "answer": 28,
     "prompt": "A car uses 8 liters of fuel per 100 km. How many liters does it need for a 350 km trip? Reply with just the number."},
]


def grade(text, answer):
    """True if the last integer in `text` equals `answer`."""
    import re
    nums = re.findall(r"-?\d[\d,]*", text or "")
    if not nums:
        return False
    try:
        return int(nums[-1].replace(",", "")) == int(answer)
    except ValueError:
        return False
