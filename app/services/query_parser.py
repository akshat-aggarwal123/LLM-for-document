import re
import spacy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ExtractedEntity:
    """Extracted entity with confidence score"""
    value: str
    confidence: float
    source: str  # 'ner', 'regex', 'keyword'

class QueryParser:
    """Parse natural language queries and extract structured information"""

    def __init__(self, settings):
        self.settings = settings
        try:
            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Regex patterns for common entities
        self.patterns = {
            'age': [
                r'\b(\d{1,3})\s*(?:year[s]?\s*old|yr[s]?\s*old|age)\b',
                r'\bage[d]?\s*(\d{1,3})\b',
                r'\b(\d{1,3})\s*y\.?o\.?\b'
            ],
            'amount': [
                r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'Rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'INR\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
            ],
            'duration': [
                r'(\d+)\s*(?:month[s]?|mon[s]?)\s*(?:old|duration)',
                r'(\d+)\s*(?:year[s]?|yr[s]?)\s*(?:old|duration|policy)',
                r'(\d+)\s*(?:day[s]?)\s*(?:old|duration)'
            ],
            'gender': [
                r'\b(male|female|man|woman|boy|girl|he|she|his|her)\b'
            ],
            'medical_procedure': [
                r'\b(surgery|operation|procedure|treatment|therapy)\b',
                r'\b(\w+\s+surgery)\b',
                r'\b(\w+\s+operation)\b'
            ]
        }

        # Medical keywords
        self.medical_keywords = {
            'procedures': [
                'surgery', 'operation', 'procedure', 'treatment', 'therapy',
                'cataract', 'knee replacement', 'heart', 'cardiac', 'orthopedic',
                'dental', 'eye', 'vision', 'hearing', 'cancer', 'oncology'
            ],
            'body_parts': [
                'eye', 'eyes', 'knee', 'heart', 'brain', 'liver', 'kidney',
                'spine', 'back', 'neck', 'shoulder', 'hip', 'ankle', 'tooth', 'teeth'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'cancer', 'tumor', 'fracture',
                'infection', 'disease', 'syndrome', 'disorder'
            ]
        }

        # Location patterns (Indian cities/states)
        self.indian_locations = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
            'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
            'maharashtra', 'karnataka', 'tamil nadu', 'gujarat', 'rajasthan',
            'west bengal', 'madhya pradesh', 'uttar pradesh', 'delhi', 'punjab'
        ]

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query and extract structured information

        Args:
            query: Natural language query string

        Returns:
            Dictionary with extracted entities
        """
        query_lower = query.lower()
        result = {
            'original_query': query,
            'entities': {},
            'confidence': 0.0,
            'medical_context': self._detect_medical_context(query_lower)
        }

        try:
            # Extract entities using different methods
            entities = {}

            # 1. Regex-based extraction
            regex_entities = self._extract_with_regex(query)
            entities.update(regex_entities)

            # 2. spaCy NER extraction
            if self.nlp:
                ner_entities = self._extract_with_ner(query)
                entities.update(ner_entities)

            # 3. Keyword-based extraction
            keyword_entities = self._extract_with_keywords(query_lower)
            entities.update(keyword_entities)

            # 4. Location extraction
            location = self._extract_location(query_lower)
            if location:
                entities['location'] = ExtractedEntity(location, 0.8, 'keyword')

            result['entities'] = {k: v.__dict__ for k, v in entities.items()}
            result['confidence'] = self._calculate_confidence(entities)

            logger.info(f"Parsed query: {len(entities)} entities extracted")

        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            result['error'] = str(e)

        return result

    def _extract_with_regex(self, query: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = {}

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)

                    # Clean and process value
                    if entity_type == 'amount':
                        value = re.sub(r'[₹$Rs\.INR\s]', '', value)
                        value = value.replace(',', '')
                    elif entity_type == 'gender':
                        value = self._normalize_gender(value)

                    entities[entity_type] = ExtractedEntity(value, 0.9, 'regex')
                    break  # Take first match for each type

        return entities

    def _extract_with_ner(self, query: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using spaCy NER"""
        entities = {}

        try:
            doc = self.nlp(query)

            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['person_name'] = ExtractedEntity(ent.text, 0.8, 'ner')
                elif ent.label_ == "GPE":  # Geopolitical entity (location)
                    entities['location'] = ExtractedEntity(ent.text, 0.8, 'ner')
                elif ent.label_ == "MONEY":
                    # Clean money value
                    value = re.sub(r'[^\d.]', '', ent.text)
                    entities['amount'] = ExtractedEntity(value, 0.8, 'ner')
                elif ent.label_ == "DATE":
                    entities['date'] = ExtractedEntity(ent.text, 0.7, 'ner')
                elif ent.label_ == "ORG":
                    entities['organization'] = ExtractedEntity(ent.text, 0.7, 'ner')

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")

        return entities

    def _extract_with_keywords(self, query: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using keyword matching"""
        entities = {}

        # Extract medical procedures
        procedures = []
        for keyword in self.medical_keywords['procedures']:
            if keyword in query:
                procedures.append(keyword)

        if procedures:
            entities['medical_procedure'] = ExtractedEntity(
                ', '.join(procedures), 0.7, 'keyword'
            )

        # Extract body parts
        body_parts = []
        for part in self.medical_keywords['body_parts']:
            if part in query:
                body_parts.append(part)

        if body_parts:
            entities['body_part'] = ExtractedEntity(
                ', '.join(body_parts), 0.7, 'keyword'
            )

        # Extract conditions
        conditions = []
        for condition in self.medical_keywords['conditions']:
            if condition in query:
                conditions.append(condition)

        if conditions:
            entities['medical_condition'] = ExtractedEntity(
                ', '.join(conditions), 0.7, 'keyword'
            )

        return entities

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract Indian locations from query"""
        for location in self.indian_locations:
            if location in query:
                return location.title()
        return None

    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender values"""
        gender = gender.lower()
        male_terms = ['male', 'man', 'boy', 'he', 'his']
        female_terms = ['female', 'woman', 'girl', 'she', 'her']

        if gender in male_terms:
            return 'Male'
        elif gender in female_terms:
            return 'Female'
        return gender.title()

    def _detect_medical_context(self, query: str) -> bool:
        """Detect if query is in medical context"""
        medical_terms = (
            self.medical_keywords['procedures'] +
            self.medical_keywords['body_parts'] +
            self.medical_keywords['conditions'] +
            ['hospital', 'doctor', 'clinic', 'medical', 'health', 'insurance', 'claim']
        )

        return any(term in query for term in medical_terms)

    def _calculate_confidence(self, entities: Dict[str, ExtractedEntity]) -> float:
        """Calculate overall confidence score"""
        if not entities:
            return 0.0

        total_confidence = sum(entity.confidence for entity in entities.values())
        avg_confidence = total_confidence / len(entities)

        # Boost confidence if multiple entities found
        entity_bonus = min(0.1 * len(entities), 0.3)

        return min(avg_confidence + entity_bonus, 1.0)

    def extract_claim_info(self, query: str) -> Dict[str, Any]:
        """
        Extract specific claim-related information

        Returns structured claim data for decision engine
        """
        parsed = self.parse_query(query)
        entities = parsed['entities']

        claim_info = {
            'patient_age': entities.get('age', {}).get('value'),
            'patient_gender': entities.get('gender', {}).get('value'),
            'procedure': entities.get('medical_procedure', {}).get('value'),
            'body_part': entities.get('body_part', {}).get('value'),
            'condition': entities.get('medical_condition', {}).get('value'),
            'location': entities.get('location', {}).get('value'),
            'claim_amount': entities.get('amount', {}).get('value'),
            'policy_duration': entities.get('duration', {}).get('value'),
            'is_medical_query': parsed['medical_context'],
            'confidence': parsed['confidence']
        }

        # Clean None values
        claim_info = {k: v for k, v in claim_info.items() if v is not None}

        logger.info(f"Extracted claim info: {claim_info}")
        return claim_info
