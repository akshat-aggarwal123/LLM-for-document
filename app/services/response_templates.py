"""Response templates for generating structured answers from document content"""

INSURANCE_QUERY_TEMPLATE = """
Based on the insurance document content, please provide a comprehensive response:

Context from document:
{context}

Question: {query}

💡 Quick Answer:
[Provide a 1-2 sentence direct answer]

📋 Detailed Explanation:
1. Key Information:
   • [Main point 1]
   • [Main point 2]
   • [Main point 3]

2. Important Requirements:
   • Eligibility Criteria: [List if applicable]
   • Documentation Needed: [List if applicable]
   • Time Limits: [Specify if any]

3. Coverage Details:
   • What's Included: [List covered items/services]
   • What's Not Included: [List exclusions]
   • Cost Information: [Copays/deductibles if mentioned]

⚠️ Important Considerations:
• [List key warnings or special conditions]
• [Include any limitations]
• [Note any exceptions]

📝 Step-by-Step Process:
1. First Step: [Action]
   • What to do
   • Required information
   • Where to go

2. Second Step: [Action]
   • Specific requirements
   • Important deadlines
   • Contact information

3. Follow-up Steps:
   • Documentation to keep
   • Next actions
   • Timeline expectations

💡 Pro Tips:
• [Practical tip 1]
• [Practical tip 2]
• [Best practice suggestion]

📞 Help Resources:
• Customer Service: [Contact details if available]
• Online Resources: [Relevant portals/websites]
• Additional Support: [Other available resources]

📄 Source Information:
• Document Section: [Section name]
• Page Number: [Page reference]
• Last Updated: [Date if available]

Note: This information is extracted directly from your insurance documentation. For specific cases or clarification, please contact your insurance provider."""

COVERAGE_EXPLANATION_TEMPLATE = """
📋 Insurance Coverage Analysis:

Document Sections Referenced:
{context}

Topic: {query}

🎯 Coverage Summary:
[One paragraph overview of the coverage]

📑 Detailed Breakdown:

1. 📌 Basic Coverage Information
   A. What's Covered:
      • [Item/Service 1]
      • [Item/Service 2]
      • [Item/Service 3]
   
   B. Key Definitions:
      • Term 1: [Definition]
      • Term 2: [Definition]
      • Term 3: [Definition]

2. 💰 Financial Information
   A. Costs:
      • Deductible: [Amount]
      • Copayment: [Amount]
      • Coinsurance: [Percentage]
   
   B. Coverage Limits:
      • Maximum Benefits: [Details]
      • Annual Limits: [Details]
      • Lifetime Limits: [Details]

3. ⚠️ Requirements and Conditions
   A. Pre-Authorization:
      • When needed
      • How to obtain
      • Timeframes
   
   B. Documentation:
      • Required forms
      • Supporting documents
      • Submission deadlines

4. 🔄 Step-by-Step Process
   A. Before Service:
      1. [First step]
      2. [Second step]
      3. [Third step]
   
   B. During Service:
      1. [What to present]
      2. [What to verify]
      3. [What to document]
   
   C. After Service:
      1. [Claim process]
      2. [Follow-up steps]
      3. [Record keeping]

5. ❌ Exclusions and Limitations
   • What's Not Covered:
     - [Exclusion 1]
     - [Exclusion 2]
     - [Exclusion 3]
   
   • Special Conditions:
     - [Condition 1]
     - [Condition 2]
     - [Condition 3]

6. 💡 Tips for Maximum Benefits
   • Do's:
     ✓ [Best practice 1]
     ✓ [Best practice 2]
     ✓ [Best practice 3]
   
   • Don'ts:
     ⛔ [What to avoid 1]
     ⛔ [What to avoid 2]
     ⛔ [What to avoid 3]

7. 🆘 Help and Support
   • Contact Information:
     - Primary: [Main contact]
     - Emergency: [Emergency contact]
     - Online: [Web resources]
   
   • When to Reach Out:
     - [Situation 1]
     - [Situation 2]
     - [Situation 3]

⚠️ Important Note: This analysis is based on the provided insurance document. Coverage details may vary based on specific circumstances. Always verify current terms and conditions with your insurance provider.

📅 Information Valid as of: [Document date]
📄 Source: [Document section/page reference]
"""

def format_response(template: str, context: str, query: str) -> str:
    """Format a response template with context and query"""
    return template.format(
        context=context,
        query=query
    )
