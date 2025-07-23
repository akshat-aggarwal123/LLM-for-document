"""
Insurance Document Analysis Model Service
Advanced language model wrapper for insurance document processing and analysis
"""

import os
import logging
from typing import Dict, List, Optional, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from ..core.config import Settings
from ..core.exceptions import LlamaModelError

logger = logging.getLogger(__name__)

class LlamaModelService:
    """Language model wrapper for insurance document analysis and response generation"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.temperature = 0.7
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Llama model with optimized settings for insurance document processing"""
        try:
            # Using a Llama model - adjust model name as needed
            model_name = "meta-llama/Llama-2-7b-chat-hf"  # or your specific Llama model
            
            logging.info(f"Initializing Llama model: {model_name} on {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.settings.MODEL_CACHE_DIR,
                model_max_length=self.max_length
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimized settings for Llama
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                cache_dir=self.settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16,  # Use float16 for Llama models
                load_in_8bit=True  # Enable 8-bit quantization for memory efficiency
            )

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info(f"Llama model initialized successfully on {self.device}")

        except Exception as e:
            error_msg = f"Failed to initialize Llama model: {str(e)}"
            logger.error(error_msg)
            raise LlamaModelError(error_msg)

    def generate_response(self, query: str, context: List[str], response_type: str = "general") -> str:
        """Generate a structured response based on the query and context"""
        try:
            # Import templates (assuming they exist)
            try:
                from .response_templates import INSURANCE_QUERY_TEMPLATE, COVERAGE_EXPLANATION_TEMPLATE
            except ImportError:
                # Fallback templates for Llama
                INSURANCE_QUERY_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert insurance analyst. Answer questions based on the provided context.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context}

Question: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                COVERAGE_EXPLANATION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an insurance coverage expert. Explain coverage details clearly and accurately.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Insurance Context: {context}

Coverage Question: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            # Combine context into a single string with proper formatting
            context_text = self._format_context(context)
            
            # Select appropriate template based on query type
            template = COVERAGE_EXPLANATION_TEMPLATE if self._is_coverage_query(query) else INSURANCE_QUERY_TEMPLATE
                
            # Create structured prompt
            prompt = template.format(context=context_text, query=query)

            # Generate response using the model
            response = self.pipeline(
                prompt,
                max_new_tokens=256,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]["generated_text"]

            return self._post_process_response(response, prompt)

        except Exception as e:
            error_msg = f"Failed to generate response: {str(e)}"
            logger.error(error_msg)
            raise LlamaModelError(error_msg)

    def _format_context(self, context: List[str]) -> str:
        """Format context passages into a structured string"""
        formatted_passages = []
        for i, passage in enumerate(context, 1):
            # Clean up the passage
            cleaned = passage.strip().replace('\n', ' ').replace('  ', ' ')
            formatted_passages.append(f"Passage {i}:\n{cleaned}\n")
        return "\n".join(formatted_passages)

    def _is_coverage_query(self, query: str) -> bool:
        """Determine if the query is about coverage explanation"""
        coverage_keywords = [
            "covered", "coverage", "benefit", "insurance", "plan",
            "what is", "how does", "explain", "mean"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in coverage_keywords)

    def _post_process_response(self, response: str, prompt: str) -> str:
        """Clean up and format the generated response"""
        # Remove the original prompt from the response
        response = response.replace(prompt, "").strip()
        
        # Remove Llama-specific tokens
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("<|end_of_text|>", "").strip()
        
        # Ensure proper formatting
        response = response.replace(" :", ":")
        response = response.replace(" ?", "?")
        response = response.replace(" .", ".")
        
        # Add line breaks for readability
        response = response.replace(". ", ".\n")
        
        return response.strip()

    def generate_decision_reasoning(
        self,
        query_data: Dict[str, Any],
        relevant_clauses: List[Dict[str, Any]],
        preliminary_decision: str
    ) -> Dict[str, Any]:
        """Generate detailed reasoning for insurance decisions"""

        try:
            prompt = self._create_decision_prompt(query_data, relevant_clauses, preliminary_decision)

            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = response[0]['generated_text']
            reasoning = self._extract_reasoning(generated_text, prompt)

            return {
                "detailed_reasoning": reasoning,
                "confidence_factors": self._extract_confidence_factors(reasoning),
                "risk_assessment": self._assess_risk(query_data, relevant_clauses)
            }

        except Exception as e:
            logger.error(f"Error generating decision reasoning: {str(e)}")
            return {
                "detailed_reasoning": f"Error in reasoning generation: {str(e)}",
                "confidence_factors": [],
                "risk_assessment": "medium"
            }

    def generate_justification(
        self,
        decision: str,
        query_data: Dict[str, Any],
        relevant_clauses: List[Dict[str, Any]],
        amount: Optional[float] = None
    ) -> str:
        """Generate human-readable justification for decisions"""

        try:
            prompt = self._create_justification_prompt(decision, query_data, relevant_clauses, amount)

            response = self.pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.05,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = response[0]['generated_text']
            justification = self._extract_justification(generated_text, prompt)

            return justification

        except Exception as e:
            logger.error(f"Error generating justification: {str(e)}")
            return f"Decision: {decision}. Unable to generate detailed justification due to processing error."

    def analyze_document_content(self, content: str, document_type: str = "insurance_policy") -> Dict[str, Any]:
        """Analyze document content for key information extraction"""

        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert insurance document analyzer. Analyze the following {document_type} content and extract key information.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Please analyze this document and provide:
1. Document type and purpose
2. Key coverage areas
3. Important exclusions
4. Waiting periods
5. Coverage limits
6. Eligibility criteria

Document Content:
{content[:2000]}...

Provide a structured analysis:

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

            generated_text = response[0]['generated_text']
            analysis = self._extract_analysis(generated_text, prompt)

            return {
                "document_summary": analysis,
                "key_sections": self._identify_key_sections(analysis),
                "coverage_areas": self._extract_coverage_areas(analysis),
                "exclusions": self._extract_exclusions(analysis)
            }

        except Exception as e:
            logger.error(f"Error analyzing document content: {str(e)}")
            return {
                "document_summary": "Error in document analysis",
                "key_sections": [],
                "coverage_areas": [],
                "exclusions": []
            }

    def _create_decision_prompt(
        self,
        query_data: Dict[str, Any],
        relevant_clauses: List[Dict[str, Any]],
        preliminary_decision: str
    ) -> str:
        """Create prompt for decision reasoning"""

        clauses_text = "\n".join([
            f"- {clause.get('content', '')[:200]}..."
            for clause in relevant_clauses[:3]
        ])

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert insurance claims analyst. Provide detailed reasoning for insurance claim decisions.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Query Details:
- Age: {query_data.get('age', 'N/A')}
- Procedure: {query_data.get('procedure', 'N/A')}
- Location: {query_data.get('location', 'N/A')}
- Policy Duration: {query_data.get('policy_duration', 'N/A')}

Relevant Policy Clauses:
{clauses_text}

Preliminary Decision: {preliminary_decision}

Please provide detailed reasoning for this decision, considering:
1. Policy compliance
2. Eligibility criteria
3. Coverage limitations
4. Risk factors
5. Precedent cases

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def _create_justification_prompt(
        self,
        decision: str,
        query_data: Dict[str, Any],
        relevant_clauses: List[Dict[str, Any]],
        amount: Optional[float]
    ) -> str:
        """Create prompt for justification generation"""

        amount_text = f"Approved Amount: ₹{amount:,.2f}" if amount else "Amount: Not applicable"

        clauses_text = "\n".join([
            f"Section {i+1}: {clause.get('content', '')[:150]}..."
            for i, clause in enumerate(relevant_clauses[:2])
        ])

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an insurance claims officer. Write a clear, professional justification for claim decisions.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Decision: {decision.upper()}
{amount_text}

Case Details:
- Patient Age: {query_data.get('age', 'N/A')}
- Medical Procedure: {query_data.get('procedure', 'N/A')}
- Treatment Location: {query_data.get('location', 'N/A')}
- Policy Age: {query_data.get('policy_duration', 'N/A')}

Applicable Policy Sections:
{clauses_text}

Write a professional justification (100-150 words) explaining this decision:

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def _extract_reasoning(self, generated_text: str, prompt: str) -> str:
        """Extract reasoning from generated text"""
        try:
            # Remove the prompt from the generated text
            reasoning = generated_text.replace(prompt, "").strip()

            # Clean up Llama-specific tokens
            if "<|eot_id|>" in reasoning:
                reasoning = reasoning.split("<|eot_id|>")[0]
            
            reasoning = reasoning.replace("<|end_of_text|>", "").strip()

            return reasoning[:1000]  # Limit length

        except Exception as e:
            logger.error(f"Error extracting reasoning: {str(e)}")
            return "Standard policy analysis applied based on available clauses."

    def _extract_justification(self, generated_text: str, prompt: str) -> str:
        """Extract justification from generated text"""
        try:
            justification = generated_text.replace(prompt, "").strip()

            # Clean up Llama-specific tokens
            if "<|eot_id|>" in justification:
                justification = justification.split("<|eot_id|>")[0]

            justification = justification.replace("<|end_of_text|>", "").strip()
            justification = justification.replace("**", "").strip()

            return justification[:500]  # Limit length

        except Exception as e:
            logger.error(f"Error extracting justification: {str(e)}")
            return "Decision made based on policy terms and coverage analysis."

    def _extract_analysis(self, generated_text: str, prompt: str) -> str:
        """Extract document analysis from generated text"""
        try:
            analysis = generated_text.replace(prompt, "").strip()

            # Clean up Llama-specific tokens
            if "<|eot_id|>" in analysis:
                analysis = analysis.split("<|eot_id|>")[0]

            analysis = analysis.replace("<|end_of_text|>", "").strip()

            return analysis[:800]

        except Exception as e:
            logger.error(f"Error extracting analysis: {str(e)}")
            return "Document analysis completed with standard parameters."

    def _extract_confidence_factors(self, reasoning: str) -> List[str]:
        """Extract confidence factors from reasoning text"""
        factors = []

        # Simple keyword-based extraction
        confidence_keywords = [
            "clear coverage", "explicit exclusion", "waiting period",
            "policy compliance", "eligibility met", "documentation complete"
        ]

        for keyword in confidence_keywords:
            if keyword.lower() in reasoning.lower():
                factors.append(keyword)

        return factors[:3]  # Return top 3 factors

    def _assess_risk(self, query_data: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> str:
        """Simple risk assessment based on query data"""

        age = query_data.get('age', 0)
        procedure = query_data.get('procedure', '').lower()

        # Simple risk scoring
        risk_score = 0

        if age > 60:
            risk_score += 1
        if 'surgery' in procedure:
            risk_score += 1
        if len(relevant_clauses) < 2:
            risk_score += 1

        if risk_score >= 2:
            return "high"
        elif risk_score == 1:
            return "medium"
        else:
            return "low"

    def _identify_key_sections(self, analysis: str) -> List[str]:
        """Identify key sections from analysis"""
        sections = []

        section_indicators = [
            "coverage", "exclusion", "waiting period",
            "eligibility", "limitation", "benefit"
        ]

        for indicator in section_indicators:
            if indicator in analysis.lower():
                sections.append(indicator.title())

        return sections[:5]

    def _extract_coverage_areas(self, analysis: str) -> List[str]:
        """Extract coverage areas from analysis"""
        coverage_areas = []

        # Simple pattern matching
        common_coverages = [
            "hospitalization", "surgery", "consultation",
            "diagnostic tests", "pharmacy", "emergency care"
        ]

        for coverage in common_coverages:
            if coverage in analysis.lower():
                coverage_areas.append(coverage.title())

        return coverage_areas

    def _extract_exclusions(self, analysis: str) -> List[str]:
        """Extract exclusions from analysis"""
        exclusions = []

        common_exclusions = [
            "pre-existing conditions", "cosmetic surgery",
            "dental treatment", "alternative medicine", "pregnancy"
        ]

        for exclusion in common_exclusions:
            if exclusion in analysis.lower():
                exclusions.append(exclusion.title())

        return exclusions