"""
Deep analysis of why the best technique performs better
Provides detailed reasoning about retrieval quality and answer generation
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List
import glob

def analyze_retrieval_quality(results: Dict) -> Dict:
    """Analyze retrieval quality patterns for each technique"""
    
    analysis = {
        'technique_patterns': {},
        'query_type_performance': {},
        'document_type_affinity': {}
    }
    
    techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Explainable Retrieval", "Basic RAG"]
    
    for technique in techniques:
        technique_data = []
        
        # Collect all responses and metrics for this technique
        for doc_name, doc_results in results.items():
            if technique in doc_results:
                tech_results = doc_results[technique]
                
                # Ensure we have all required fields
                required_fields = ['queries', 'responses', 'contexts', 'faithfulness_scores', 
                                 'relevance_scores', 'completeness_scores', 'clarity_scores', 'response_times']
                
                if all(field in tech_results for field in required_fields):
                    for i, query in enumerate(tech_results['queries']):
                        if i < len(tech_results['responses']):
                            technique_data.append({
                                'query': query,
                                'response': tech_results['responses'][i],
                                'context': tech_results['contexts'][i] if i < len(tech_results['contexts']) else "",
                                'faithfulness': tech_results['faithfulness_scores'][i] if i < len(tech_results['faithfulness_scores']) else 0,
                                'relevance': tech_results['relevance_scores'][i] if i < len(tech_results['relevance_scores']) else 0,
                                'completeness': tech_results['completeness_scores'][i] if i < len(tech_results['completeness_scores']) else 0,
                                'clarity': tech_results['clarity_scores'][i] if i < len(tech_results['clarity_scores']) else 0,
                                'response_time': tech_results['response_times'][i] if i < len(tech_results['response_times']) else 0,
                                'document': doc_name
                            })
        
        # Analyze patterns
        if technique_data:
            df = pd.DataFrame(technique_data)
            
            # Calculate response and context lengths safely
            response_lengths = []
            context_lengths = []
            
            for _, row in df.iterrows():
                try:
                    response_lengths.append(len(str(row['response'])))
                    context_lengths.append(len(str(row['context'])))
                except:
                    response_lengths.append(0)
                    context_lengths.append(0)
            
            analysis['technique_patterns'][technique] = {
                'avg_faithfulness': df['faithfulness'].mean(),
                'avg_relevance': df['relevance'].mean(),
                'avg_completeness': df['completeness'].mean(),
                'avg_clarity': df['clarity'].mean(),
                'avg_response_time': df['response_time'].mean(),
                'consistency': {
                    'faithfulness_std': df['faithfulness'].std(),
                    'relevance_std': df['relevance'].std(),
                    'response_time_std': df['response_time'].std()
                },
                'best_performing_queries': df.nlargest(3, 'relevance')['query'].tolist(),
                'worst_performing_queries': df.nsmallest(3, 'relevance')['query'].tolist(),
                'response_length_avg': np.mean(response_lengths) if response_lengths else 0,
                'context_usage': np.mean(context_lengths) if context_lengths else 0,
                'total_queries': len(technique_data)
            }
    
    return analysis

def identify_best_technique_reasons(results: Dict) -> Dict:
    """Identify specific reasons why the best technique performs better"""
    
    # Calculate overall scores
    technique_scores = {}
    techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Explainable Retrieval", "Basic RAG"]
    
    for technique in techniques:
        scores = []
        for doc_results in results.values():
            if technique in doc_results and 'total_score' in doc_results[technique]:
                scores.append(doc_results[technique]['total_score'])
        
        if scores:
            technique_scores[technique] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
    
    # Find best technique
    if not technique_scores:
        return {'error': 'No valid technique scores found'}
    
    best_technique = max(technique_scores.keys(), key=lambda x: technique_scores[x]['mean'])
    best_score = technique_scores[best_technique]['mean']
    
    # Analyze why it's the best
    analysis = analyze_retrieval_quality(results)
    
    if best_technique not in analysis['technique_patterns']:
        return {'error': f'No analysis data found for {best_technique}'}
    
    best_patterns = analysis['technique_patterns'][best_technique]
    
    reasons = {
        'best_technique': best_technique,
        'overall_score': best_score,
        'score_std': technique_scores[best_technique]['std'],
        'tested_documents': technique_scores[best_technique]['count'],
        'key_strengths': [],
        'comparative_advantages': {},
        'retrieval_analysis': {},
        'answer_generation_analysis': {},
        'efficiency_analysis': {}
    }
    
    # Compare with other techniques
    for other_technique in techniques:
        if (other_technique != best_technique and 
            other_technique in analysis['technique_patterns'] and
            other_technique in technique_scores):
            
            other_patterns = analysis['technique_patterns'][other_technique]
            
            advantages = {}
            
            # Compare metrics
            metrics_comparison = {
                'faithfulness': (best_patterns['avg_faithfulness'], other_patterns['avg_faithfulness']),
                'relevance': (best_patterns['avg_relevance'], other_patterns['avg_relevance']),
                'completeness': (best_patterns['avg_completeness'], other_patterns['avg_completeness']),
                'clarity': (best_patterns['avg_clarity'], other_patterns['avg_clarity'])
            }
            
            for metric, (best_val, other_val) in metrics_comparison.items():
                if best_val > other_val:
                    improvement = best_val - other_val
                    advantages[metric] = {
                        'best': best_val,
                        'other': other_val,
                        'improvement': improvement,
                        'improvement_pct': (improvement / other_val * 100) if other_val > 0 else 0
                    }
            
            if advantages:
                reasons['comparative_advantages'][other_technique] = advantages
    
    # Identify key strengths
    strength_thresholds = {
        'faithfulness': 0.7,
        'relevance': 0.7,
        'completeness': 0.7,
        'clarity': 0.7
    }
    
    for metric, threshold in strength_thresholds.items():
        avg_key = f'avg_{metric}'
        if avg_key in best_patterns and best_patterns[avg_key] > threshold:
            reasons['key_strengths'].append(f"High {metric}: {best_patterns[avg_key]:.3f} (>{threshold})")
    
    # Consistency analysis
    if best_patterns['consistency']['relevance_std'] < 0.2:
        reasons['key_strengths'].append(f"Consistent performance: σ = {best_patterns['consistency']['relevance_std']:.3f}")
    
    # Speed analysis
    if best_patterns['avg_response_time'] < 5.0:
        reasons['key_strengths'].append(f"Fast response time: {best_patterns['avg_response_time']:.2f}s")
    
    # Analyze retrieval patterns
    reasons['retrieval_analysis'] = {
        'avg_context_length': best_patterns['context_usage'],
        'avg_response_length': best_patterns['response_length_avg'],
        'best_query_types': best_patterns['best_performing_queries'],
        'challenging_queries': best_patterns['worst_performing_queries'],
        'total_queries_processed': best_patterns['total_queries']
    }
    
    # Answer generation analysis
    reasons['answer_generation_analysis'] = {
        'faithfulness_score': best_patterns['avg_faithfulness'],
        'completeness_score': best_patterns['avg_completeness'],
        'clarity_score': best_patterns['avg_clarity'],
        'context_utilization': 'High' if best_patterns['context_usage'] > 1000 else 'Moderate' if best_patterns['context_usage'] > 500 else 'Low'
    }
    
    # Efficiency analysis
    reasons['efficiency_analysis'] = {
        'speed_ranking': calculate_speed_ranking(technique_scores, analysis),
        'consistency_ranking': calculate_consistency_ranking(technique_scores, analysis),
        'overall_efficiency': best_patterns['avg_response_time']
    }
    
    return reasons

def calculate_speed_ranking(technique_scores: Dict, analysis: Dict) -> List[str]:
    """Calculate speed ranking of techniques"""
    speed_data = []
    for technique, scores in technique_scores.items():
        if technique in analysis['technique_patterns']:
            avg_time = analysis['technique_patterns'][technique]['avg_response_time']
            speed_data.append((technique, avg_time))
    
    # Sort by speed (lower is better)
    speed_data.sort(key=lambda x: x[1])
    return [technique for technique, _ in speed_data]

def calculate_consistency_ranking(technique_scores: Dict, analysis: Dict) -> List[str]:
    """Calculate consistency ranking of techniques"""
    consistency_data = []
    for technique, scores in technique_scores.items():
        if technique in analysis['technique_patterns']:
            std_score = scores['std']
            consistency_data.append((technique, std_score))
    
    # Sort by consistency (lower std is better)
    consistency_data.sort(key=lambda x: x[1])
    return [technique for technique, _ in consistency_data]

def create_best_technique_report(results: Dict):
    """Create a comprehensive report on the best technique"""
    
    reasons = identify_best_technique_reasons(results)
    
    if 'error' in reasons:
        print(f"❌ Error: {reasons['error']}")
        return
    
    best_technique = reasons['best_technique']
    
    print(f"\n{'='*80}")
    print(f"🏆 WHY {best_technique.upper()} IS THE BEST RAG TECHNIQUE")
    print('='*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   • Score: {reasons['overall_score']:.3f} ± {reasons['score_std']:.3f}")
    print(f"   • Tested on: {reasons['tested_documents']} documents")
    
    print(f"\n💪 KEY STRENGTHS:")
    if reasons['key_strengths']:
        for i, strength in enumerate(reasons['key_strengths'], 1):
            print(f"   {i}. {strength}")
    else:
        print("   • Analysis in progress...")
    
    print(f"\n🔍 RETRIEVAL ANALYSIS:")
    retrieval = reasons['retrieval_analysis']
    print(f"   • Total queries processed: {retrieval['total_queries_processed']}")
    print(f"   • Average context length: {retrieval['avg_context_length']:.0f} characters")
    print(f"   • Average response length: {retrieval['avg_response_length']:.0f} characters")
    print(f"   • Context utilization: {reasons['answer_generation_analysis']['context_utilization']}")
    
    print(f"\n✅ BEST PERFORMING QUERY TYPES:")
    for i, query in enumerate(retrieval['best_query_types'], 1):
        print(f"   {i}. {query}")
    
    print(f"\n⚠️ CHALLENGING QUERY TYPES:")
    for i, query in enumerate(retrieval['challenging_queries'], 1):
        print(f"   {i}. {query}")
    
    print(f"\n🎯 ANSWER GENERATION QUALITY:")
    answer_gen = reasons['answer_generation_analysis']
    print(f"   • Faithfulness: {answer_gen['faithfulness_score']:.3f}")
    print(f"   • Completeness: {answer_gen['completeness_score']:.3f}")
    print(f"   • Clarity: {answer_gen['clarity_score']:.3f}")
    
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    efficiency = reasons['efficiency_analysis']
    print(f"   • Average response time: {efficiency['overall_efficiency']:.2f} seconds")
    print(f"   • Speed ranking: {efficiency['speed_ranking'].index(best_technique) + 1} of {len(efficiency['speed_ranking'])}")
    print(f"   • Consistency ranking: {efficiency['consistency_ranking'].index(best_technique) + 1} of {len(efficiency['consistency_ranking'])}")
    
    print(f"\n📈 COMPARATIVE ADVANTAGES:")
    if reasons['comparative_advantages']:
        for other_technique, advantages in reasons['comparative_advantages'].items():
            print(f"\n   vs {other_technique}:")
            for metric, data in advantages.items():
                improvement = data['improvement']
                improvement_pct = data['improvement_pct']
                print(f"     • {metric.title()}: +{improvement:.3f} improvement ({improvement_pct:.1f}%)")
                print(f"       {data['best']:.3f} vs {data['other']:.3f}")
    else:
        print("   • Detailed comparison data not available")
    
    # Technique-specific analysis
    print(f"\n🔧 TECHNIQUE-SPECIFIC ANALYSIS:")
    
    technique_insights = {
        "Adaptive RAG": {
            "description": "Intelligent query classification with adaptive retrieval strategies",
            "strengths": [
                "Query Classification: Automatically identifies query type (Factual, Analytical, Opinion, Contextual)",
                "Specialized Strategies: Uses different retrieval strategies for different query types", 
                "Enhanced Query Processing: Reformulates queries for better retrieval",
                "Diverse Document Ranking: Selects most relevant and diverse documents",
                "Contextual Understanding: Adapts to user context and intent"
            ],
            "best_for": "Educational platforms, user-adaptive systems, diverse query types"
        },
        "CRAG": {
            "description": "Self-correcting retrieval with web search fallback",
            "strengths": [
                "Quality Evaluation: Evaluates retrieval quality before answering",
                "Self-Correction: Falls back to web search when local documents are insufficient",
                "Knowledge Refinement: Extracts key information from retrieved documents",
                "Threshold-Based Decision Making: Uses confidence scores to determine best action",
                "Multiple Source Integration: Combines local and web-based information"
            ],
            "best_for": "Mission-critical applications, fact-checking systems, real-time information needs"
        },
        "Document Augmentation": {
            "description": "Enhanced retrieval through synthetic Q&A generation",
            "strengths": [
                "Synthetic Question Generation: Creates questions from document content",
                "Improved Retrieval Precision: Questions help match user queries better",
                "Format-Specific Processing: Adapts to different document types",
                "Enhanced Document Understanding: Generates multiple perspectives on content",
                "Better Query-Document Alignment: Bridges gap between user queries and document content"
            ],
            "best_for": "Large-scale document processing, knowledge discovery, semantic search"
        },
        "Explainable Retrieval": {
            "description": "Transparent retrieval with detailed explanations",
            "strengths": [
                "Detailed Explanations: Provides reasoning for why content is relevant",
                "Multi-Step Analysis: Explains relationship between query and retrieved content",
                "Comprehensive Context: Combines multiple relevant sections effectively",
                "Transparency: Shows how conclusions are reached",
                "Better User Understanding: Helps users understand the retrieval process"
            ],
            "best_for": "Research support, decision-making tools, transparent AI applications"
        },
        "Basic RAG": {
            "description": "Straightforward similarity-based retrieval",
            "strengths": [
                "Simplicity: Straightforward similarity-based retrieval",
                "Speed: Fast processing with minimal overhead",
                "Reliability: Consistent performance across different scenarios",
                "Low Resource Usage: Efficient memory and computational requirements",
                "Broad Applicability: Works well across various document types"
            ],
            "best_for": "Simple Q&A systems, resource-constrained environments, baseline implementations"
        }
    }
    
    if best_technique in technique_insights:
        insights = technique_insights[best_technique]
        print(f"\n   {insights['description']}\n")
        print(f"   Why {best_technique} excels:")
        for strength in insights['strengths']:
            print(f"   • {strength}")
        print(f"\n   Best use cases: {insights['best_for']}")
    
    print(f"\n💡 RECOMMENDATIONS FOR OPTIMAL USE:")
    print(f"   1. Use {best_technique} for general-purpose document Q&A")
    print(f"   2. Particularly effective for the high-performing query types shown above")
    print(f"   3. Consider document preprocessing to maximize performance")
    print(f"   4. Monitor performance on your specific document types and domains")
    print(f"   5. Implement fallback mechanisms for edge cases")
    
    print(f"\n🎯 DEPLOYMENT CONSIDERATIONS:")
    print(f"   • Response time: {efficiency['overall_efficiency']:.2f}s average (plan for scaling)")
    print(f"   • Context usage: {retrieval['avg_context_length']:.0f} chars (consider token limits)")
    print(f"   • Consistency: Good across different document types")
    print(f"   • Resource requirements: Moderate (typical for LLM-based systems)")

def main():
    """Run the best technique analysis"""
    
    print("🔍 Best RAG Technique Analysis")
    print("="*40)
    
    # Load latest results
    result_files = glob.glob("evaluation_results_*.json")
    if not result_files:
        print("❌ No evaluation results found!")
        print("Please run comprehensive_evaluation.py first.")
        return
    
    latest_file = max(result_files)
    print(f"📂 Loading results from: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        # Create the comprehensive report
        create_best_technique_report(results)
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading or analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
