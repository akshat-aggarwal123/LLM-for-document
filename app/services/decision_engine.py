import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
from app.utils.logging import get_logger

logger = get_logger(__name__)

class DecisionType(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class DecisionResult:
    decision: DecisionType
    confidence_score: float
    confidence_level: ConfidenceLevel
    approved_amount: Optional[float]
    justification: str
    relevant_clauses: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    processing_time: float

@dataclass
class PolicyRule:
    name: str
    condition: str
    action: str
    priority: int
    confidence_impact: float

class DecisionEngine:
    """AI-powered decision engine for insurance claims"""

    def __init__(self, settings):
        self.settings = settings
        self.policy_rules = self._load_default_rules()
        self.risk_thresholds = {
            'high_risk_age': 70,
            'max_claim_amount': 500000,
            'min_policy_duration_months': 24,
            'high_risk_procedures': [
                'heart surgery', 'brain surgery', 'cancer treatment',
                'organ transplant', 'cardiac procedure'
            ]
        }

        # Common coverage limits and exclusions
        self.coverage_limits = {
            'eye_surgery': 150000,
            'dental': 25000,
            'maternity': 100000,
            'orthopedic': 200000,
            'cardiac': 300000,
            'cancer': 500000,
            'general_surgery': 100000
        }

        self.exclusions = [
            'cosmetic surgery', 'experimental treatment', 'pre-existing condition',
            'self-inflicted injury', 'substance abuse', 'war injury'
        ]

    def _load_default_rules(self) -> List[PolicyRule]:
        """Load default policy decision rules"""
        return [
            PolicyRule(
                name="Age Verification",
                condition="age_valid",
                action="approve_if_true",
                priority=1,
                confidence_impact=0.2
            ),
            PolicyRule(
                name="Policy Duration Check",
                condition="policy_duration_sufficient",
                action="approve_if_true",
                priority=2,
                confidence_impact=0.3
            ),
            PolicyRule(
                name="Coverage Limit Check",
                condition="within_coverage_limit",
                action="approve_if_true",
                priority=3,
                confidence_impact=0.3
            ),
            PolicyRule(
                name="Exclusion Check",
                condition="not_excluded_procedure",
                action="approve_if_true",
                priority=4,
                confidence_impact=0.4
            ),
            PolicyRule(
                name="Pre-authorization Check",
                condition="pre_auth_obtained",
                action="approve_if_true",
                priority=5,
                confidence_impact=0.2
            )
        ]

    def make_decision(self,
                     claim_info: Dict[str, Any],
                     retrieved_documents: List[Dict[str, Any]]) -> DecisionResult:
        """
        Make a decision on insurance claim based on claim info and policy documents

        Args:
            claim_info: Extracted claim information
            retrieved_documents: Relevant policy documents

        Returns:
            DecisionResult with decision and justification
        """
        start_time = datetime.now()

        try:
            # Initialize decision components
            decision_scores = {}
            risk_factors = []
            relevant_clauses = []
            recommendations = []

            # 1. Analyze claim against policy documents
            policy_analysis = self._analyze_policy_documents(claim_info, retrieved_documents)
            decision_scores.update(policy_analysis['scores'])
            relevant_clauses.extend(policy_analysis['clauses'])

            # 2. Apply business rules
            rule_analysis = self._apply_business_rules(claim_info)
            decision_scores.update(rule_analysis['scores'])
            risk_factors.extend(rule_analysis['risk_factors'])

            # 3. Check exclusions
            exclusion_check = self._check_exclusions(claim_info, retrieved_documents)
            if exclusion_check['is_excluded']:
                processing_time = (datetime.now() - start_time).total_seconds()
                return DecisionResult(
                    decision=DecisionType.REJECTED,
                    confidence_score=0.9,
                    confidence_level=ConfidenceLevel.HIGH,
                    approved_amount=0.0,
                    justification=exclusion_check['reason'],
                    relevant_clauses=exclusion_check['clauses'],
                    risk_factors=risk_factors,
                    recommendations=["Review policy exclusions"],
                    processing_time=processing_time
                )

            # 4. Calculate coverage amount
            approved_amount = self._calculate_approved_amount(claim_info, retrieved_documents)

            # 5. Determine final decision
            overall_score = self._calculate_overall_score(decision_scores)
            decision_type = self._determine_decision_type(overall_score, risk_factors)
            confidence_level = self._determine_confidence_level(overall_score, len(retrieved_documents))

            # 6. Generate justification
            justification = self._generate_justification(
                decision_type, claim_info, decision_scores, relevant_clauses
            )

            # 7. Generate recommendations
            recommendations = self._generate_recommendations(
                decision_type, claim_info, risk_factors
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            return DecisionResult(
                decision=decision_type,
                confidence_score=overall_score,
                confidence_level=confidence_level,
                approved_amount=approved_amount if decision_type == DecisionType.APPROVED else 0.0,
                justification=justification,
                relevant_clauses=relevant_clauses,
                risk_factors=risk_factors,
                recommendations=recommendations,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            # Default error case decision
            return DecisionResult(
                decision=DecisionType.REQUIRES_REVIEW,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.LOW,
                approved_amount=0.0,
                justification=f"Error in automated processing: {str(e)}. Manual review required.",
                relevant_clauses=[],
                risk_factors=["Processing Error"],
                recommendations=["Manual review required"],
                processing_time=processing_time
            )

    def _analyze_policy_documents(self, claim_info: Dict[str, Any],
                                documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retrieved policy documents against claim"""
        scores = {}
        relevant_clauses = []

        if not documents:
            return {'scores': {'document_coverage': 0.0}, 'clauses': []}

        # Analyze document relevance and coverage
        coverage_indicators = []
        exclusion_indicators = []

        for doc in documents:
            content = doc['content'].lower()
            similarity = doc.get('similarity_score', 0.0)

            # Look for coverage indicators
            if any(term in content for term in ['covered', 'benefits', 'eligible', 'included']):
                coverage_indicators.append(similarity)
                relevant_clauses.append(f"Coverage clause: {content[:100]}...")

            # Look for exclusion indicators
            if any(term in content for term in ['excluded', 'not covered', 'limitations', 'restricted']):
                exclusion_indicators.append(similarity)
                relevant_clauses.append(f"Exclusion clause: {content[:100]}...")

        # Calculate coverage score
        if coverage_indicators:
            coverage_score = sum(coverage_indicators) / len(coverage_indicators)
        else:
            coverage_score = 0.3  # Default low coverage if no clear indicators

        # Reduce score if exclusions found
        if exclusion_indicators:
            exclusion_penalty = sum(exclusion_indicators) / len(exclusion_indicators) * 0.5
            coverage_score = max(0.0, coverage_score - exclusion_penalty)

        scores['document_coverage'] = coverage_score
        scores['document_relevance'] = sum(doc.get('similarity_score', 0) for doc in documents) / len(documents)

        return {'scores': scores, 'clauses': relevant_clauses}

    def _apply_business_rules(self, claim_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business rules to claim"""
        scores = {}
        risk_factors = []

        # Age-based risk assessment
        if claim_info.get('patient_age'):
            age = int(claim_info['patient_age'])
            if age > self.risk_thresholds['high_risk_age']:
                scores['age_risk'] = 0.6
                risk_factors.append(f"High risk age: {age} years")
            elif age < 18:
                scores['age_risk'] = 0.7
                risk_factors.append("Pediatric case - requires special consideration")
            else:
                scores['age_risk'] = 0.9

        # Claim amount validation
        if claim_info.get('claim_amount'):
            amount = float(claim_info['claim_amount'])
            if amount > self.risk_thresholds['max_claim_amount']:
                scores['amount_risk'] = 0.4
                risk_factors.append(f"High claim amount: ₹{amount:,.2f}")
            else:
                scores['amount_risk'] = 0.8

        # Policy duration check
        if claim_info.get('policy_duration'):
            duration_str = claim_info['policy_duration']
            duration_months = self._parse_duration_to_months(duration_str)

            if duration_months < self.risk_thresholds['min_policy_duration_months']:
                scores['policy_duration'] = 0.5
                risk_factors.append(f"Short policy duration: {duration_months} months")
            else:
                scores['policy_duration'] = 0.9

        # Procedure risk assessment
        if claim_info.get('procedure'):
            procedure = claim_info['procedure'].lower()
            if any(high_risk in procedure for high_risk in self.risk_thresholds['high_risk_procedures']):
                scores['procedure_risk'] = 0.6
                risk_factors.append(f"High-risk procedure: {claim_info['procedure']}")
            else:
                scores['procedure_risk'] = 0.8

        return {'scores': scores, 'risk_factors': risk_factors}

    def _check_exclusions(self, claim_info: Dict[str, Any],
                         documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if claim falls under policy exclusions"""

        # Check against common exclusions
        if claim_info.get('procedure'):
            procedure = claim_info['procedure'].lower()

            for exclusion in self.exclusions:
                if exclusion in procedure:
                    return {
                        'is_excluded': True,
                        'reason': f"Procedure '{claim_info['procedure']}' is excluded under policy terms: {exclusion}",
                        'clauses': [f"Exclusion: {exclusion}"]
                    }

        # Check document-based exclusions
        for doc in documents:
            content = doc['content'].lower()

            # Look for specific exclusion patterns
            if 'not covered' in content or 'excluded' in content:
                if claim_info.get('procedure'):
                    procedure = claim_info['procedure'].lower()
                    # Simple keyword matching for exclusions
                    if any(word in content for word in procedure.split()):
                        return {
                            'is_excluded': True,
                            'reason': f"Procedure may be excluded based on policy document",
                            'clauses': [content[:200] + "..."]
                        }

        return {'is_excluded': False, 'reason': '', 'clauses': []}

    def _calculate_approved_amount(self, claim_info: Dict[str, Any],
                                 documents: List[Dict[str, Any]]) -> float:
        """Calculate approved claim amount"""
        requested_amount = 0.0

        if claim_info.get('claim_amount'):
            requested_amount = float(claim_info['claim_amount'])

        # Determine coverage limit based on procedure
        coverage_limit = None
        if claim_info.get('procedure'):
            procedure = claim_info['procedure'].lower()

            for coverage_type, limit in self.coverage_limits.items():
                if coverage_type in procedure:
                    coverage_limit = limit
                    break

        # Default coverage limit if no specific limit found
        if coverage_limit is None:
            coverage_limit = self.coverage_limits['general_surgery']

        # Return minimum of requested amount and coverage limit
        if requested_amount > 0:
            return min(requested_amount, coverage_limit)
        else:
            return coverage_limit

    def _calculate_overall_score(self, decision_scores: Dict[str, float]) -> float:
        """Calculate overall decision score"""
        if not decision_scores:
            return 0.0

        # Weighted average of all scores
        weights = {
            'document_coverage': 0.3,
            'document_relevance': 0.2,
            'age_risk': 0.15,
            'amount_risk': 0.1,
            'policy_duration': 0.15,
            'procedure_risk': 0.1
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for score_name, score_value in decision_scores.items():
            weight = weights.get(score_name, 0.1)  # Default weight
            weighted_sum += score_value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_decision_type(self, overall_score: float,
                               risk_factors: List[str]) -> DecisionType:
        """Determine final decision type"""

        # High risk factors override score
        high_risk_count = len([rf for rf in risk_factors if 'high' in rf.lower()])

        if high_risk_count > 2:
            return DecisionType.REQUIRES_REVIEW

        # Score-based decision
        if overall_score >= 0.8:
            return DecisionType.APPROVED
        elif overall_score >= 0.6:
            return DecisionType.REQUIRES_REVIEW
        elif overall_score >= 0.4:
            return DecisionType.REQUIRES_REVIEW
        else:
            return DecisionType.REJECTED

    def _determine_confidence_level(self, score: float, doc_count: int) -> ConfidenceLevel:
        """Determine confidence level"""

        # Adjust confidence based on available documentation
        doc_factor = min(1.0, doc_count / 5.0)  # Normalize to 5 docs
        adjusted_score = score * doc_factor

        if adjusted_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif adjusted_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_justification(self, decision: DecisionType, claim_info: Dict[str, Any],
                              scores: Dict[str, float], clauses: List[str]) -> str:
        """Generate human-readable justification"""

        justification_parts = []

        # Decision statement
        if decision == DecisionType.APPROVED:
            justification_parts.append("Claim APPROVED based on policy analysis.")
        elif decision == DecisionType.REJECTED:
            justification_parts.append("Claim REJECTED due to policy violations.")
        elif decision == DecisionType.REQUIRES_REVIEW:
            justification_parts.append("Claim marked as PENDING - additional information required.")
        else:
            justification_parts.append("Claim requires MANUAL REVIEW due to complexity.")

        # Add specific reasons
        if claim_info.get('procedure'):
            justification_parts.append(f"Procedure: {claim_info['procedure']}")

        if claim_info.get('patient_age'):
            justification_parts.append(f"Patient age: {claim_info['patient_age']} years")

        # Add score-based reasoning
        if scores.get('document_coverage', 0) > 0.7:
            justification_parts.append("Strong policy coverage support found.")
        elif scores.get('document_coverage', 0) < 0.4:
            justification_parts.append("Limited policy coverage support.")

        # Add relevant clauses
        if clauses:
            justification_parts.append("Key policy references:")
            for clause in clauses[:3]:  # Limit to top 3
                justification_parts.append(f"- {clause}")

        return " ".join(justification_parts)

    def _generate_recommendations(self, decision: DecisionType, claim_info: Dict[str, Any],
                                risk_factors: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if decision == DecisionType.REQUIRES_REVIEW:
            recommendations.append("Obtain additional medical documentation")
            recommendations.append("Verify policy coverage details")

        if decision == DecisionType.REQUIRES_REVIEW:
            recommendations.append("Schedule manual review with claims specialist")
            recommendations.append("Consider pre-authorization requirements")

        if decision == DecisionType.REJECTED:
            recommendations.append("Review policy terms with customer")
            recommendations.append("Explore alternative coverage options")

        # Risk-based recommendations
        if any('high risk' in rf.lower() for rf in risk_factors):
            recommendations.append("Consider additional medical evaluation")

        if any('amount' in rf.lower() for rf in risk_factors):
            recommendations.append("Review claim amount justification")

        return recommendations

    def _parse_duration_to_months(self, duration_str: str) -> int:
        """Parse duration string to months"""
        try:
            duration_str = duration_str.lower()

            # Extract number
            number_match = re.search(r'(\d+)', duration_str)
            if not number_match:
                return 0

            number = int(number_match.group(1))

            # Determine unit
            if 'year' in duration_str:
                return number * 12
            elif 'month' in duration_str:
                return number
            elif 'day' in duration_str:
                return max(1, number // 30)  # Approximate
            else:
                return number  # Assume months

        except:
            return 0

    def get_decision_explanation(self, decision_result: DecisionResult) -> Dict[str, Any]:
        """Get detailed explanation of decision"""
        return {
            'decision_summary': {
                'final_decision': decision_result.decision.value,
                'confidence_score': decision_result.confidence_score,
                'confidence_level': decision_result.confidence_level.value,
                'processing_time': f"{decision_result.processing_time:.2f} seconds"
            },
            'financial_details': {
                'approved_amount': decision_result.approved_amount,
                'currency': 'INR'
            },
            'supporting_evidence': {
                'relevant_clauses': decision_result.relevant_clauses,
                'justification': decision_result.justification
            },
            'risk_assessment': {
                'risk_factors': decision_result.risk_factors,
                'recommendations': decision_result.recommendations
            }
        }
