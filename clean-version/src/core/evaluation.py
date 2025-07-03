"""
Evaluation system for assessing retrieval quality and answer accuracy.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import statistics
from datetime import datetime

from ..llm.gemini import GeminiLLM
from ..utils.exceptions import EvaluationError
from ..utils.helpers import calculate_similarity, clean_text
from config.prompts import EVALUATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluation results."""
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    confidence_score: float
    overall_score: float
    feedback: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalEvaluation:
    """Evaluation of retrieval results."""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    metrics: EvaluationMetrics
    timestamp: datetime


class EvaluationSystem:
    """System for evaluating retrieval and generation quality."""
    
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        self.evaluation_history = []
        
    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        ground_truth: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate the quality of retrieved documents.
        
        Args:
            query: The original query
            retrieved_documents: List of retrieved documents
            ground_truth: Optional list of known correct documents
            
        Returns:
            EvaluationMetrics with scores and feedback
        """
        try:
            logger.info(f"Evaluating retrieval for query: {query[:100]}...")
            
            # Calculate relevance scores
            relevance_scores = []
            for doc in retrieved_documents:
                content = doc.get('content', '')
                relevance = await self._evaluate_relevance(query, content)
                relevance_scores.append(relevance)
            
            # Calculate completeness
            completeness_score = await self._evaluate_completeness(query, retrieved_documents)
            
            # Calculate accuracy if ground truth is available
            accuracy_score = 0.8  # Default when no ground truth
            if ground_truth:
                accuracy_score = self._calculate_accuracy(retrieved_documents, ground_truth)
            
            # Calculate confidence based on consistency
            confidence_score = self._calculate_confidence(relevance_scores)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                relevance_scores, completeness_score, accuracy_score, confidence_score
            )
            
            # Generate feedback
            feedback = await self._generate_feedback(
                query, retrieved_documents, relevance_scores, completeness_score
            )
            
            metrics = EvaluationMetrics(
                relevance_score=statistics.mean(relevance_scores) if relevance_scores else 0.0,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                confidence_score=confidence_score,
                overall_score=overall_score,
                feedback=feedback,
                metadata={
                    'num_documents': len(retrieved_documents),
                    'individual_relevance_scores': relevance_scores,
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            )
            
            # Store evaluation
            evaluation = RetrievalEvaluation(
                query=query,
                retrieved_documents=retrieved_documents,
                metrics=metrics,
                timestamp=datetime.now()
            )
            self.evaluation_history.append(evaluation)
            
            logger.info(f"Evaluation completed. Overall score: {overall_score:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise EvaluationError(f"Evaluation failed: {str(e)}")
    
    async def _evaluate_relevance(self, query: str, content: str) -> float:
        """Evaluate relevance of content to query."""
        try:
            # Use LLM for detailed relevance evaluation
            prompt = f"""
            Evaluate how relevant the following content is to the query.
            Provide a score from 0.0 to 1.0 where 1.0 is perfectly relevant.
            
            Query: {query}
            Content: {content[:1000]}
            
            Respond with just the numeric score (e.g., 0.8):
            """
            
            response = await self.llm.generate_response(prompt)
            
            # Extract score from response
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                # Fallback to simple similarity
                return calculate_similarity(query, content)
                
        except Exception as e:
            logger.warning(f"LLM relevance evaluation failed: {str(e)}")
            return calculate_similarity(query, content)
    
    async def _evaluate_completeness(self, query: str, documents: List[Dict[str, Any]]) -> float:
        """Evaluate completeness of retrieved documents."""
        try:
            combined_content = "\n\n".join([doc.get('content', '') for doc in documents])
            
            prompt = f"""
            Evaluate how completely the following content answers the query.
            Provide a score from 0.0 to 1.0 where 1.0 means the query is fully answered.
            
            Query: {query}
            Combined Content: {combined_content[:2000]}
            
            Respond with just the numeric score (e.g., 0.7):
            """
            
            response = await self.llm.generate_response(prompt)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                # Fallback calculation
                return min(1.0, len(documents) / 5.0)  # Assume 5 docs for completeness
                
        except Exception as e:
            logger.warning(f"Completeness evaluation failed: {str(e)}")
            return min(1.0, len(documents) / 5.0)
    
    def _calculate_accuracy(self, retrieved_docs: List[Dict[str, Any]], ground_truth: List[str]) -> float:
        """Calculate accuracy against ground truth."""
        if not ground_truth:
            return 0.8  # Default when no ground truth
        
        # Simple implementation - in practice, this would be more sophisticated
        retrieved_content = [doc.get('content', '') for doc in retrieved_docs]
        
        matches = 0
        for truth in ground_truth:
            for content in retrieved_content:
                if calculate_similarity(truth, content) > 0.7:
                    matches += 1
                    break
        
        return matches / len(ground_truth) if ground_truth else 0.0
    
    def _calculate_confidence(self, relevance_scores: List[float]) -> float:
        """Calculate confidence based on consistency of relevance scores."""
        if not relevance_scores:
            return 0.0
        
        if len(relevance_scores) == 1:
            return relevance_scores[0]
        
        # High confidence if scores are consistently high
        mean_score = statistics.mean(relevance_scores)
        std_dev = statistics.stdev(relevance_scores)
        
        # Confidence is higher when mean is high and std dev is low
        confidence = mean_score * (1 - min(std_dev, 0.5))
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_overall_score(
        self,
        relevance_scores: List[float],
        completeness_score: float,
        accuracy_score: float,
        confidence_score: float
    ) -> float:
        """Calculate overall evaluation score."""
        relevance_score = statistics.mean(relevance_scores) if relevance_scores else 0.0
        
        # Weighted combination
        overall = (
            relevance_score * 0.4 +
            completeness_score * 0.3 +
            accuracy_score * 0.2 +
            confidence_score * 0.1
        )
        
        return max(0.0, min(1.0, overall))
    
    async def _generate_feedback(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        relevance_scores: List[float],
        completeness_score: float
    ) -> str:
        """Generate human-readable feedback."""
        try:
            avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
            
            prompt = f"""
            Generate concise feedback for a document retrieval evaluation.
            
            Query: {query}
            Number of documents: {len(documents)}
            Average relevance score: {avg_relevance:.2f}
            Completeness score: {completeness_score:.2f}
            
            Provide specific feedback on:
            1. Quality of retrieved documents
            2. Completeness of information
            3. Suggestions for improvement
            
            Keep it concise (2-3 sentences):
            """
            
            feedback = await self.llm.generate_response(prompt)
            return feedback.strip()
            
        except Exception as e:
            logger.warning(f"Feedback generation failed: {str(e)}")
            return f"Retrieved {len(documents)} documents with average relevance of {avg_relevance:.2f}"
    
    async def evaluate_answer(
        self,
        query: str,
        answer: str,
        source_documents: List[Dict[str, Any]],
        ground_truth_answer: Optional[str] = None
    ) -> EvaluationMetrics:
        """Evaluate the quality of a generated answer."""
        try:
            logger.info(f"Evaluating answer for query: {query[:100]}...")
            
            # Evaluate factual accuracy
            accuracy_score = await self._evaluate_answer_accuracy(answer, source_documents)
            
            # Evaluate completeness
            completeness_score = await self._evaluate_answer_completeness(query, answer)
            
            # Evaluate relevance
            relevance_score = await self._evaluate_relevance(query, answer)
            
            # Calculate confidence
            confidence_score = min(accuracy_score, completeness_score, relevance_score)
            
            # Compare with ground truth if available
            if ground_truth_answer:
                ground_truth_similarity = calculate_similarity(answer, ground_truth_answer)
                accuracy_score = (accuracy_score + ground_truth_similarity) / 2
            
            overall_score = (accuracy_score + completeness_score + relevance_score) / 3
            
            feedback = await self._generate_answer_feedback(
                query, answer, accuracy_score, completeness_score, relevance_score
            )
            
            return EvaluationMetrics(
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                confidence_score=confidence_score,
                overall_score=overall_score,
                feedback=feedback,
                metadata={
                    'answer_length': len(answer),
                    'num_source_documents': len(source_documents),
                    'has_ground_truth': ground_truth_answer is not None,
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Answer evaluation failed: {str(e)}")
            raise EvaluationError(f"Answer evaluation failed: {str(e)}")
    
    async def _evaluate_answer_accuracy(self, answer: str, source_documents: List[Dict[str, Any]]) -> float:
        """Evaluate accuracy of answer against source documents."""
        try:
            combined_sources = "\n\n".join([doc.get('content', '') for doc in source_documents])
            
            prompt = f"""
            Evaluate how accurately the answer is supported by the source documents.
            Provide a score from 0.0 to 1.0 where 1.0 means fully accurate.
            
            Answer: {answer}
            Source Documents: {combined_sources[:2000]}
            
            Respond with just the numeric score (e.g., 0.9):
            """
            
            response = await self.llm.generate_response(prompt)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.7  # Default score
                
        except Exception as e:
            logger.warning(f"Answer accuracy evaluation failed: {str(e)}")
            return 0.7
    
    async def _evaluate_answer_completeness(self, query: str, answer: str) -> float:
        """Evaluate completeness of answer."""
        try:
            prompt = f"""
            Evaluate how completely the answer addresses the query.
            Provide a score from 0.0 to 1.0 where 1.0 means fully complete.
            
            Query: {query}
            Answer: {answer}
            
            Respond with just the numeric score (e.g., 0.8):
            """
            
            response = await self.llm.generate_response(prompt)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.7  # Default score
                
        except Exception as e:
            logger.warning(f"Answer completeness evaluation failed: {str(e)}")
            return 0.7
    
    async def _generate_answer_feedback(
        self,
        query: str,
        answer: str,
        accuracy_score: float,
        completeness_score: float,
        relevance_score: float
    ) -> str:
        """Generate feedback for answer evaluation."""
        try:
            prompt = f"""
            Generate concise feedback for an answer evaluation.
            
            Query: {query}
            Answer: {answer[:500]}...
            Accuracy: {accuracy_score:.2f}
            Completeness: {completeness_score:.2f}
            Relevance: {relevance_score:.2f}
            
            Provide specific feedback on strengths and areas for improvement.
            Keep it concise (2-3 sentences):
            """
            
            feedback = await self.llm.generate_response(prompt)
            return feedback.strip()
            
        except Exception as e:
            logger.warning(f"Answer feedback generation failed: {str(e)}")
            return f"Answer quality: Accuracy {accuracy_score:.2f}, Completeness {completeness_score:.2f}, Relevance {relevance_score:.2f}"
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about evaluations."""
        if not self.evaluation_history:
            return {
                'total_evaluations': 0,
                'average_scores': {},
                'recent_evaluations': []
            }
        
        recent_evaluations = self.evaluation_history[-10:]  # Last 10 evaluations
        
        # Calculate average scores
        avg_relevance = statistics.mean([e.metrics.relevance_score for e in recent_evaluations])
        avg_completeness = statistics.mean([e.metrics.completeness_score for e in recent_evaluations])
        avg_accuracy = statistics.mean([e.metrics.accuracy_score for e in recent_evaluations])
        avg_confidence = statistics.mean([e.metrics.confidence_score for e in recent_evaluations])
        avg_overall = statistics.mean([e.metrics.overall_score for e in recent_evaluations])
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_scores': {
                'relevance': avg_relevance,
                'completeness': avg_completeness,
                'accuracy': avg_accuracy,
                'confidence': avg_confidence,
                'overall': avg_overall
            },
            'recent_evaluations': [
                {
                    'query': e.query[:100],
                    'overall_score': e.metrics.overall_score,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in recent_evaluations
            ]
        }
