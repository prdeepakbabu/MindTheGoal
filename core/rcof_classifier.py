"""RCOF (Root Cause of Failure) Classifier implementation."""

from typing import Optional, Dict, List
from core.models import RCOF, Turn, TurnQuality


class RCOFClassifier:
    """
    Classifier for Root Cause of Failure (RCOF) taxonomy.
    
    Maps failure patterns to RCOF codes E1-E7 based on the paper's taxonomy.
    """
    
    # Keywords/patterns associated with each RCOF category
    RCOF_PATTERNS: Dict[str, List[str]] = {
        "E1": [  # Language Understanding Failure
            "misunderstood", "didn't understand", "wrong interpretation",
            "misinterpreted", "confused", "incorrect context"
        ],
        "E2": [  # Refusal to Answer
            "i cannot", "i can't", "i'm unable", "not allowed",
            "refuse to", "won't be able", "cannot help with"
        ],
        "E3": [  # Incorrect Retrieval
            "wrong information", "incorrect data", "outdated",
            "inaccurate", "different from what", "not what you asked"
        ],
        "E4": [  # Retrieval Failure
            "couldn't find", "no information", "not found",
            "no results", "unable to locate", "no data available"
        ],
        "E5": [  # System Error
            "error occurred", "system error", "timed out",
            "technical issue", "failed to process", "service unavailable"
        ],
        "E6": [  # Incorrect Routing
            "wrong department", "wrong team", "not my area",
            "different service", "redirect to", "transferred"
        ],
        "E7": [  # Out-of-Domain
            "outside my capabilities", "not supported", "beyond scope",
            "cannot perform", "not designed for", "not available"
        ]
    }

    def classify(
        self, 
        turn: Turn,
        judge_reasoning: Optional[str] = None,
        explicit_code: Optional[str] = None
    ) -> Optional[str]:
        """
        Classify a failed turn into an RCOF category.
        
        Priority:
        1. Explicit code from judge (if provided)
        2. Pattern matching on judge reasoning
        3. Pattern matching on response content
        4. Default to E1 (Language Understanding) if no match
        
        Args:
            turn: The turn to classify
            judge_reasoning: Optional reasoning from LLM judge
            explicit_code: Optional explicit RCOF code from judge
            
        Returns:
            RCOF code (E1-E7) or None if turn is successful
        """
        if turn.quality != TurnQuality.FAILURE:
            return None
        
        # Priority 1: Explicit code from judge
        if explicit_code and self.is_valid_code(explicit_code):
            return explicit_code
        
        # Priority 2: Pattern match on judge reasoning
        if judge_reasoning:
            code = self._match_patterns(judge_reasoning)
            if code:
                return code
        
        # Priority 3: Pattern match on response content
        if turn.agent_response:
            code = self._match_patterns(turn.agent_response)
            if code:
                return code
        
        # Default: E1 (Language Understanding) as most common failure
        return "E1"

    def _match_patterns(self, text: str) -> Optional[str]:
        """
        Match text against RCOF patterns.
        
        Args:
            text: Text to search for patterns
            
        Returns:
            RCOF code if match found, None otherwise
        """
        text_lower = text.lower()
        
        for code, patterns in self.RCOF_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    return code
        
        return None

    def is_valid_code(self, code: str) -> bool:
        """Check if a code is a valid RCOF code."""
        return code in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

    def get_description(self, code: str) -> str:
        """Get description for an RCOF code."""
        return RCOF.get_description(code)

    def get_all_codes(self) -> Dict[str, str]:
        """Get all RCOF codes with descriptions."""
        return RCOF.get_all_descriptions()

    def analyze_distribution(
        self, 
        turns: List[Turn]
    ) -> Dict[str, Dict]:
        """
        Analyze RCOF distribution across failed turns.
        
        Args:
            turns: List of turns to analyze
            
        Returns:
            Distribution with counts and percentages
        """
        failed_turns = [t for t in turns if t.quality == TurnQuality.FAILURE]
        total_failures = len(failed_turns)
        
        if total_failures == 0:
            return {
                "total_failures": 0,
                "distribution": {},
                "percentages": {}
            }
        
        # Count by RCOF code
        counts = {}
        for turn in failed_turns:
            code = turn.rcof or "E1"
            counts[code] = counts.get(code, 0) + 1
        
        # Calculate percentages
        percentages = {
            code: (count / total_failures) * 100
            for code, count in counts.items()
        }
        
        return {
            "total_failures": total_failures,
            "distribution": counts,
            "percentages": percentages
        }
