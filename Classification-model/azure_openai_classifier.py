"""LLM Few-Shot Document Classifier using Azure OpenAI."""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


@dataclass
class ClassificationResult:
    domain: str
    category: str
    confidence: float
    reasoning: str
    tokens_used: int
    cost_usd: float
    error: Optional[str] = None


class LLMClassifier:
    """Document classifier using Azure OpenAI with few-shot prompting."""
    
    CATEGORIES = {
        "Law": {
            "Contracts": "Agreements, terms, parties, obligations, clauses, signatures (commercial, employment, lease, NDA)",
            "Litigation": "Lawsuits, claims, plaintiffs, defendants, petitions, appeals, motions, complaints",
            "Court Rulings": "Judgments, verdicts, court decisions, judicial opinions, sentences, orders",
            "Legislation": "Statutes, regulations, codes, legislative acts, amendments, articles of law",
            "Legal Opinions": "Advisory opinions, legal memos, counsel guidance, fatwas, attorney guidance",
        }
    }
    
    FEW_SHOT_EXAMPLES = """
Examples (analyze domain AND sub-category keywords):

[Law > Contracts] "وقّع الطرفان عقد شراكة تجارية وفقاً لأحكام القانون المدني" → Keywords: عقد، طرفان، شراكة
[Law > Contracts] "The parties executed a non-disclosure agreement with standard indemnification clauses" → Keywords: agreement, parties, clauses, executed

[Law > Litigation] "رفع المحامي دعوى تعويض أمام المحكمة الابتدائية ضد الشركة" → Keywords: محامي، دعوى، محكمة، تعويض
[Law > Litigation] "The plaintiff filed a class-action lawsuit alleging securities fraud and breach of fiduciary duty" → Keywords: plaintiff, lawsuit, complaint, filed

[Law > Court Rulings] "أصدرت المحكمة الدستورية العليا حكماً بعدم دستورية المادة الثانية من القانون" → Keywords: محكمة، حكم، دستورية، مادة
[Law > Court Rulings] "The Supreme Court delivered a landmark ruling on patent eligibility and prior art" → Keywords: Court, ruling, judgment, Supreme Court

[Law > Legislation] "نشرت الجريدة الرسمية قانون العمل الجديد رقم 12 لسنة 2024 المعدل للائحة" → Keywords: قانون، لائحة، رسمية، مادة
[Law > Legislation] "The Data Protection Act 2024 introduces new compliance requirements for processors" → Keywords: Act, regulation, compliance, law

[Law > Legal Opinions] "أصدر مجلس الدولة فتوى قانونية بشأن نزاع عقود التعيينات الحكومية" → Keywords: فتوى، قانونية، مجلس، استشارة
[Law > Legal Opinions] "The Attorney General issued a formal legal opinion on the constitutionality of the proposed treaty" → Keywords: legal opinion, Attorney General, counsel

[Uncertain] "The weather is nice today and the sun is shining" → No legal keywords
[Uncertain] "I had breakfast this morning with friends" → Generic, no category fit
"""
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize LLM classifier.
        
        Args:
            provider: Only "azure" supported
            api_key: API key (or set AZURE_OPENAI_API_KEY env var)
            model: Not used (uses deployment_name)
            endpoint: Azure OpenAI endpoint
            deployment_name: Azure deployment name (e.g., "gpt-4o-mini")
            api_version: Azure API version (default: 2024-12-01-preview)
        """
        self.provider = provider.lower()
        
        if self.provider == "azure":
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.model = self.deployment_name  
            
            if not all([self.api_key, self.endpoint, self.deployment_name]):
                raise ValueError(
                    "Missing Azure OpenAI configuration. Set environment variables:\n"
                    "- AZURE_OPENAI_API_KEY\n"
                    "- AZURE_OPENAI_ENDPOINT\n"
                    "- AZURE_OPENAI_DEPLOYMENT_NAME"
                )
            
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )
            
            self.input_cost = 0.00015
            self.output_cost = 0.0006
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Only 'azure' is supported.")
    
    def _extract_smart_excerpt(self, document: str, max_tokens: int = 1200) -> str:
        lines = [line.strip() for line in document.split('\n') if line.strip()]
        total_lines = len(lines)
        
        if total_lines <= 50:
            return document
        
        first_page_count = min(30, total_lines // 5)
        first_page = lines[:first_page_count]
        
        last_page_count = min(15, total_lines // 10)
        last_page = lines[-last_page_count:] if last_page_count > 0 else []
        
        signal_keywords = [
            'invoice', 'contract', 'agreement', 'total', 'amount', 'salary',
            'payment', 'report', 'summary', 'conclusion', 'recommendation',
            'فاتورة', 'عقد', 'اتفاقية', 'المجموع', 'المبلغ', 'راتب',
            'دفع', 'تقرير', 'ملخص', 'خلاصة', 'توصية', 'قرار'
        ]
        
        middle_start = first_page_count
        middle_end = total_lines - last_page_count
        high_signal_lines = []
        
        for i in range(middle_start, middle_end):
            line = lines[i]
            has_numbers = any(char.isdigit() for char in line)
            has_keyword = any(kw in line.lower() for kw in signal_keywords)
            is_title_like = len(line) < 100 and line.endswith(':')
            
            if (has_numbers or has_keyword or is_title_like) and len(line) > 10:
                high_signal_lines.append(line)
        
        middle_excerpt = high_signal_lines[:25]
        
        excerpt_parts = []
        excerpt_parts.extend(first_page)
        
        if middle_excerpt:
            excerpt_parts.append("")
            excerpt_parts.append("..." * 20)
            excerpt_parts.append("")
            excerpt_parts.extend(middle_excerpt)
        
        if last_page:
            excerpt_parts.append("")
            excerpt_parts.append("..." * 20)
            excerpt_parts.append("")
            excerpt_parts.extend(last_page)
        
        return '\n'.join(excerpt_parts)
    
    def _build_prompt(self, text: str, is_excerpt: bool = False) -> str:
        categories_text = ""
        for domain_name, sub_cats in self.CATEGORIES.items():
            for cat, desc in sub_cats.items():
                categories_text += f"- {domain_name} > {cat} ({desc})\n"
        
        if is_excerpt:
            purpose_guidance = """
FOCUS ON DOCUMENT PURPOSE (not just keywords mentioned):
- "Article about legal AI tools for contract review" → Law > Technology-related but the document is ABOUT legal work → Law > Legal Opinions
- "News article about a famous court case" → Law > Court Rulings (document describes a ruling)
- "Contract template for hospital equipment purchase" → Law > Contracts (it's a contract, the hospital context is incidental)
"""
        else:
            purpose_guidance = ""
        
        prompt = f"""You are an expert legal document classifier for Arabic and English documents.

Classify the following document into ONE domain and ONE sub-category:
{categories_text}

{self.FEW_SHOT_EXAMPLES}
{purpose_guidance}
CLASSIFICATION STRATEGY:
1. Check if the document primarily concerns legal matters (عقد→Contracts, محكمة→Litigation/Rulings, قانون→Legislation, فتوى→Legal Opinions, court→Rulings, contract→Contracts, etc.)
2. Identify the specific legal sub-category based on content
3. If the document clearly falls under Law but the sub-category is ambiguous, pick the most likely one
4. Only use "Uncertain" if the document has NO legal content whatsoever (weather, personal chat, recipes, etc.)
5. Confidence: 70-90% for clear cases, 50-70% for ambiguous ones

Document to classify:
\"\"\"{text}\"\"\"

Respond ONLY with a JSON object in this EXACT format (no additional text before or after):
{{"domain": "Law", "category": "CategoryName", "confidence": 0.XX, "reasoning": "Mention KEY WORDS that led to decision"}}

Rules:
- domain: Must be one of: Law
- category: Must be one of: Contracts, Litigation, Court Rulings, Legislation, Legal Opinions, Uncertain
- confidence: Number between 0.0 and 1.0 (be realistic - not everything is 95%)
- reasoning: Brief explanation (1-2 sentences) citing specific keywords
- Output ONLY JSON object

IMPORTANT: For legal documents, always use domain "Law". Only use category "Uncertain" if the document is NOT legal at all!"""
        
        return prompt
    
    def classify(
        self,
        text: str,
        max_retries: int = 3,
        use_excerpt: bool = True,
        excerpt_threshold: int = 2000
    ) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                domain="Error",
                category="Error",
                confidence=0.0,
                reasoning="Empty document provided",
                tokens_used=0,
                cost_usd=0.0,
                error="Empty input"
            )
        
        original_length = len(text)
        is_long_doc = original_length > excerpt_threshold
        
        if use_excerpt and is_long_doc:
            text = self._extract_smart_excerpt(text)
            is_excerpt = True
        else:
            is_excerpt = False
        
        prompt = self._build_prompt(text, is_excerpt=is_excerpt)
        
        for attempt in range(max_retries):
            try:
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert document classifier. Respond ONLY with valid JSON. Analyze keywords carefully and classify with confidence."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_completion_tokens": 200,
                }
                
                response = self.client.chat.completions.create(**request_params)
                
                content = response.choices[0].message.content.strip()
                
                # Debug: print the raw response
                print(f"[DEBUG] Raw response: {content[:200]}")
                
                try:
                    result_dict = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    
                    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_block:
                        result_dict = json.loads(json_block.group(1))
                    else:
                        json_match = re.search(r'\{[^{]*?"category"[^}]*?\}', content, re.DOTALL)
                        if json_match:
                            try:
                                result_dict = json.loads(json_match.group(0))
                            except:
                                result_dict = self._parse_text_response(content)
                        else:
                            result_dict = self._parse_text_response(content)
                
                tokens_used = response.usage.total_tokens
                cost_usd = (
                    (response.usage.prompt_tokens / 1000) * self.input_cost +
                    (response.usage.completion_tokens / 1000) * self.output_cost
                )
                
                domain = result_dict.get("domain", "Law")
                if domain not in self.CATEGORIES:
                    domain = "Law"
                
                category = result_dict.get("category", "Uncertain")
                if domain in self.CATEGORIES and category not in self.CATEGORIES[domain]:
                    category = "Uncertain"
                
                return ClassificationResult(
                    domain=domain,
                    category=category,
                    confidence=float(result_dict.get("confidence", 0.5)),
                    reasoning=result_dict.get("reasoning", ""),
                    tokens_used=tokens_used,
                    cost_usd=cost_usd
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {e}"
                if attempt == max_retries - 1:
                    return ClassificationResult(
                        domain="Error",
                        category="Error",
                        confidence=0.0,
                        reasoning=error_msg,
                        tokens_used=0,
                        cost_usd=0.0,
                        error=error_msg
                    )
                time.sleep(2 ** attempt)
                
            except Exception as e:
                error_msg = f"Classification failed: {str(e)}"
                if attempt == max_retries - 1:
                    return ClassificationResult(
                        domain="Error",
                        category="Error",
                        confidence=0.0,
                        reasoning=error_msg,
                        tokens_used=0,
                        cost_usd=0.0,
                        error=error_msg
                    )
                time.sleep(2 ** attempt)
        
        return ClassificationResult(
            domain="Error",
            category="Error",
            confidence=0.0,
            reasoning="Max retries exceeded",
            tokens_used=0,
            cost_usd=0.0,
            error="Max retries exceeded"
        )
    
    def _parse_text_response(self, content: str) -> Dict:
        import re
        
        domain = "Law"
        category = "Uncertain"
        confidence = 0.5
        
        content_lower = content.lower()
        
        all_sub_cats = []
        for domain_name, sub_cats in self.CATEGORIES.items():
            for cat in sub_cats.keys():
                all_sub_cats.append(cat)
        
        for cat in all_sub_cats:
            patterns = [
                rf'\b{cat.lower()}\b',
                rf'\[{cat.lower()}\]',
                rf'category["\s:]*{cat.lower()}',
            ]
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    category = cat
                    break
            if category != "Uncertain":
                break
        
        dom_match = re.search(r'domain["\s:]*["\']?([A-Za-z]+)["\']?', content)
        if dom_match:
            domain = dom_match.group(1).capitalize()
            if domain not in self.CATEGORIES:
                domain = "Law"
        
        conf_patterns = [
            r'"confidence"["\s:]+(\d+\.?\d*)',
            r'confidence["\s:]+(\d+\.?\d*)',
            r'confidence.*?(\d+\.?\d*)\s*[,}]',
        ]
        
        confidence_found = False
        for pattern in conf_patterns:
            match = re.search(pattern, content_lower)
            if match:
                conf_value = float(match.group(1))
                confidence = conf_value if conf_value <= 1 else conf_value / 100
                confidence_found = True
                break
        
        if not confidence_found:
            if category in content:
                confidence = 0.75
            else:
                confidence = 0.5
        
        reasoning_match = re.search(r'reasoning["\s:]+["\']([^"\']+)["\']', content, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
        else:
            reasoning = content[:200].strip()
        
        return {
            "domain": domain,
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def _find_closest_category(self, category: str) -> str:
        category_lower = category.lower()
        for domain_name, sub_cats in self.CATEGORIES.items():
            for valid_cat in sub_cats.keys():
                if category_lower in valid_cat.lower() or valid_cat.lower() in category_lower:
                    return valid_cat
        return "Uncertain"
    
    def classify_batch(
        self,
        texts: List[str],
        delay: float = 0.5,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Classify multiple documents sequentially with rate limiting.
        
        Args:
            texts: List of document texts
            delay: Seconds to wait between requests (rate limiting)
            show_progress: Show progress bar
            
        Returns:
            DataFrame with classification results
        """
        results = []
        iterator = tqdm(texts, desc="Classifying") if show_progress else texts
        
        for text in iterator:
            result = self.classify(text)
            results.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'domain': result.domain,
                'category': result.category,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'tokens_used': result.tokens_used,
                'cost_usd': result.cost_usd,
                'error': result.error
            })
            
            time.sleep(delay)
        
        df = pd.DataFrame(results)
        
        if show_progress:
            self._print_summary(df)
        
        return df
    
    def evaluate_against_labels(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> Dict:
        """
        Classify documents and compare against true labels.
        
        Args:
            df: DataFrame with documents and labels
            text_column: Name of column containing text
            label_column: Name of column containing true labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        predictions = []
        true_labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            result = self.classify(row[text_column])
            predictions.append(result.category)
            true_labels.append(row[label_column])
            time.sleep(0.5)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return results
