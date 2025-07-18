import re
from dataclasses import dataclass
from typing import Optional
import spacy

@dataclass
class QueryEntity:
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    amount: Optional[float] = None
    raw_query: str = ""

class QueryParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

        self.patterns = {
            "age": r"\b(\d{1,2})\s*(?:yrs?|years?|y\.o\.?)\b",
            "gender": r"\b(?:male|female|man|woman|m|f)\b",
            "amount": r"(?:₹|rs\.?|inr|\$)\s*([\d,]+(?:\.\d+)?)",
            "location": r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "duration": r"(\d+)\s*(?:months?|years?)\s*(?:policy|old)?",
        }

    def extract_entities(self, query: str) -> QueryEntity:
        q = query.lower()
        ent = QueryEntity(raw_query=query)

        # regex passes
        age_m = re.search(self.patterns["age"], q)
        if age_m:
            ent.age = int(age_m.group(1))

        gender_m = re.search(self.patterns["gender"], q)
        if gender_m:
            g = gender_m.group().upper()
            ent.gender = "Male" if g in {"M", "MALE", "MAN"} else "Female"

        amt_m = re.search(self.patterns["amount"], query, re.I)
        if amt_m:
            ent.amount = float(amt_m.group(1).replace(",", ""))

        loc_m = re.search(self.patterns["location"], query)
        if loc_m:
            ent.location = loc_m.group(1)

        dur_m = re.search(self.patterns["duration"], q)
        if dur_m:
            ent.policy_duration = dur_m.group()

        # spaCy extras
        if self.nlp:
            doc = self.nlp(query)
            for e in doc.ents:
                if e.label_ == "GPE" and not ent.location:
                    ent.location = e.text
                elif e.label_ == "MONEY" and not ent.amount:
                    num = re.search(r"[\d,]+", e.text)
                    if num:
                        ent.amount = float(num.group().replace(",", ""))

        return ent