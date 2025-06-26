"""
Visualize RAG evaluation results with comprehensive charts and analysis
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

def load_latest_results():
    """Load the most recent evaluation results"""
    result_files = glob.glob("evaluation_results_*.json")
    if not result_files:
        print("❌ No evaluation results found!")
        print("Please run comprehensive_evaluation.py first.")
        return None
    
    latest_file = max(result_files)
    print(f"📂 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_comprehensive_visualizations(results):
    """Create comprehensive visualizations of RAG evaluation results"""
    
    # Prepare data for visualization
    techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Explainable Retrieval", "Basic RAG"]
    metrics = ['avg_faithfulness', 'avg_relevance', 'avg_completeness', 'avg_clarity']
    metric_names = ['Faithfulness', 'Relevance', 'Completeness', 'Clarity']
    
    # Collect data across all documents
    viz_data = []
    for doc_name, doc_results in results.items():
        for technique in techniques:
            if technique in doc_results and 'total_score' in doc_results[technique]:
                row = {
                    'Document': doc_name,
                    'Technique': technique,
                    'Total_Score': doc_results[technique]['total_score'],
                    'Response_Time': doc_results[technique]['avg_response_time']
                }
                for metric, name in zip(metrics, metric_names):
                    if metric in doc_results[technique]:
                        row[name] = doc_results[technique][metric]
                    else:
                        row[name] = 0  # Default value
                viz_data.append(row)
    
    if not viz_data:
        print("❌ No data available for visualization!")
        return
    
    df = pd.DataFrame(viz_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('RAG Technique Comprehensive Evaluation Results', fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors for techniques
    colors = {
        'Adaptive RAG': '#2E8B57',
        'CRAG': '#4169E1', 
        'Document Augmentation': '#DC143C',
        'Explainable Retrieval': '#FF8C00',
        'Basic RAG': '#9932CC'
    }
    
    # 1. Overall Performance Comparison (Top Left)
    plt.subplot(3, 3, 1)
    technique_scores = df.groupby('Technique')['Total_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bars = plt.bar(range(len(technique_scores)), technique_scores['mean'], 
                   yerr=technique_scores['std'], capsize=5,
                   color=[colors.get(tech, '#888888') for tech in technique_scores.index])
    
    plt.title('Overall Performance Ranking', fontsize=14, fontweight='bold')
    plt.ylabel('Average Total Score')
    plt.xticks(range(len(technique_scores)), technique_scores.index, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, technique_scores['mean'], technique_scores['std'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.02, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Metric Breakdown Heatmap (Top Center)
    plt.subplot(3, 3, 2)
    heatmap_data = df.groupby('Technique')[metric_names].mean()
    sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
                cbar_kws={'label': 'Score'}, fmt='.3f')
    plt.title('Performance by Metric', fontsize=14, fontweight='bold')
    plt.ylabel('Metrics')
    
    # 3. Response Time Comparison (Top Right)
    plt.subplot(3, 3, 3)
    response_times = df.groupby('Technique')['Response_Time'].agg(['mean', 'std']).sort_values('mean')
    bars = plt.bar(range(len(response_times)), response_times['mean'],
                   yerr=response_times['std'], capsize=5,
                   color=[colors.get(tech, '#888888') for tech in response_times.index])
    
    plt.title('Average Response Time', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(response_times)), response_times.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, response_times['mean'], response_times['std'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1, 
                f'{mean_val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Performance by Document Type (Middle Left)
    plt.subplot(3, 3, 4)
    doc_performance = df.pivot_table(values='Total_Score', index='Document', 
                                   columns='Technique', aggfunc='mean', fill_value=0)
    
    # Create grouped bar chart
    x = np.arange(len(doc_performance.index))
    width = 0.15
    
    for i, technique in enumerate(doc_performance.columns):
        if technique in colors:
            plt.bar(x + i*width, doc_performance[technique], width, 
                   label=technique, color=colors[technique], alpha=0.8)
    
    plt.title('Performance by Document Type', fontsize=14, fontweight='bold')
    plt.ylabel('Total Score')
    plt.xlabel('Documents')
    plt.xticks(x + width*2, [doc.replace('.txt', '').replace('.csv', '').replace('.json', '')[:15] 
                             for doc in doc_performance.index], rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 5. Faithfulness vs Relevance Scatter (Middle Center)
    plt.subplot(3, 3, 5)
    for technique in techniques:
        if technique in df['Technique'].values:
            tech_data = df[df['Technique'] == technique]
            if not tech_data.empty and 'Faithfulness' in tech_data.columns and 'Relevance' in tech_data.columns:
                plt.scatter(tech_data['Faithfulness'], tech_data['Relevance'], 
                           c=colors.get(technique, '#888888'), label=technique, alpha=0.7, s=100)
    
    plt.xlabel('Faithfulness')
    plt.ylabel('Relevance')
    plt.title('Faithfulness vs Relevance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 6. Score Distribution Box Plot (Middle Right)
    plt.subplot(3, 3, 6)
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for technique in technique_scores.index:  # Use sorted order
        tech_scores = df[df['Technique'] == technique]['Total_Score'].dropna()
        if not tech_scores.empty:
            box_data.append(tech_scores)
            box_labels.append(technique)
    
    if box_data:
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, technique in zip(bp['boxes'], box_labels):
            patch.set_facecolor(colors.get(technique, '#888888'))
            patch.set_alpha(0.7)
    
    plt.title('Score Distribution by Technique', fontsize=14, fontweight='bold')
    plt.ylabel('Total Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 7. Radar Chart for Top 3 Techniques (Bottom Left)
    plt.subplot(3, 3, 7, projection='polar')
    top_3_techniques = technique_scores.head(3).index
    
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, technique in enumerate(top_3_techniques):
        if technique in heatmap_data.index:
            values = heatmap_data.loc[technique].tolist()
            values += values[:1]  # Complete the circle
            
            color = colors.get(technique, '#888888')
            plt.plot(angles, values, 'o-', linewidth=2, label=technique, color=color)
            plt.fill(angles, values, alpha=0.25, color=color)
    
    plt.xticks(angles[:-1], metric_names)
    plt.title('Top 3 Techniques Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.ylim(0, 1)
    
    # 8. Metric Correlation Matrix (Bottom Center)
    plt.subplot(3, 3, 8)
    available_columns = ['Total_Score', 'Response_Time'] + [col for col in metric_names if col in df.columns]
    correlation_data = df[available_columns].select_dtypes(include=[np.number])
    
    if not correlation_data.empty:
        corr_matrix = correlation_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                    square=True, cbar_kws={'label': 'Correlation'}, fmt='.2f')
    
    plt.title('Metric Correlations', fontsize=14, fontweight='bold')
    
    # 9. Performance Consistency (Bottom Right)
    plt.subplot(3, 3, 9)
    
    # Calculate coefficient of variation (std/mean) for each technique
    consistency_data = []
    for technique in techniques:
        tech_data = df[df['Technique'] == technique]['Total_Score']
        if not tech_data.empty and tech_data.mean() > 0:
            cv = tech_data.std() / tech_data.mean()
            consistency_data.append((technique, cv, tech_data.mean()))
    
    if consistency_data:
        consistency_df = pd.DataFrame(consistency_data, columns=['Technique', 'CV', 'Mean'])
        consistency_df = consistency_df.sort_values('CV')  # Lower CV = more consistent
        
        bars = plt.bar(range(len(consistency_df)), consistency_df['CV'],
                       color=[colors.get(tech, '#888888') for tech in consistency_df['Technique']])
        
        plt.title('Performance Consistency\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
        plt.ylabel('Coefficient of Variation')
        plt.xticks(range(len(consistency_df)), consistency_df['Technique'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cv in zip(bars, consistency_df['CV']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{cv:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig('rag_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive visualization saved as 'rag_evaluation_comprehensive.png'")
    
    plt.show()
    
    return df, technique_scores

def print_detailed_analysis(df, technique_scores):
    """Print detailed text analysis of results"""
    print("\n" + "="*80)
    print("📊 DETAILED ANALYSIS REPORT")
    print("="*80)
    
    # Best technique overall
    best_technique = technique_scores.index[0]
    best_score = technique_scores.loc[best_technique, 'mean']
    best_std = technique_scores.loc[best_technique, 'std']
    
    print(f"\n🏆 BEST OVERALL TECHNIQUE: {best_technique}")
    print(f"   📈 Average Score: {best_score:.3f} ± {best_std:.3f}")
    
    # Performance breakdown for best technique
    best_metrics = df[df['Technique'] == best_technique].agg({
        'Faithfulness': ['mean', 'std'],
        'Relevance': ['mean', 'std'], 
        'Completeness': ['mean', 'std'],
        'Clarity': ['mean', 'std'],
        'Response_Time': ['mean', 'std']
    })
    
    print(f"\n💪 DETAILED PERFORMANCE OF {best_technique}:")
    for metric in ['Faithfulness', 'Relevance', 'Completeness', 'Clarity']:
        if metric in best_metrics.columns:
            mean_val = best_metrics[metric]['mean']
            std_val = best_metrics[metric]['std']
            print(f"   {metric}: {mean_val:.3f} ± {std_val:.3f}")
    
    if 'Response_Time' in best_metrics.columns:
        time_mean = best_metrics['Response_Time']['mean']
        time_std = best_metrics['Response_Time']['std']
        print(f"   Response Time: {time_mean:.2f} ± {time_std:.2f} seconds")
    
    # Speed analysis
    speed_ranking = df.groupby('Technique')['Response_Time'].mean().sort_values()
    fastest_technique = speed_ranking.index[0]
    fastest_time = speed_ranking.iloc[0]
    print(f"\n⚡ FASTEST TECHNIQUE: {fastest_technique} ({fastest_time:.2f}s average)")
    
    # Consistency analysis
    consistency_data = []
    for technique in df['Technique'].unique():
        tech_data = df[df['Technique'] == technique]['Total_Score']
        if len(tech_data) > 1:
            cv = tech_data.std() / tech_data.mean() if tech_data.mean() > 0 else float('inf')
            consistency_data.append((technique, cv))
    
    if consistency_data:
        consistency_df = pd.DataFrame(consistency_data, columns=['Technique', 'CV']).sort_values('CV')
        most_consistent = consistency_df.iloc[0]['Technique']
        consistency_score = consistency_df.iloc[0]['CV']
        print(f"\n🎯 MOST CONSISTENT: {most_consistent} (CV = {consistency_score:.3f})")
    
    # Document type affinity
    print(f"\n📄 DOCUMENT TYPE PERFORMANCE:")
    doc_performance = df.groupby(['Document', 'Technique'])['Total_Score'].mean().unstack(fill_value=0)
    
    for doc in doc_performance.index:
        best_for_doc = doc_performance.loc[doc].idxmax()
        best_score_for_doc = doc_performance.loc[doc].max()
        print(f"   {doc}: {best_for_doc} ({best_score_for_doc:.3f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use {best_technique} for general-purpose document Q&A")
    print(f"   2. Use {fastest_technique} when speed is critical")
    if consistency_data:
        print(f"   3. Use {most_consistent} when consistency is important")
    print(f"   4. Consider document type when choosing technique")
    
    # Strengths and weaknesses
    print(f"\n🔍 TECHNIQUE ANALYSIS:")
    for technique in technique_scores.index[:3]:  # Top 3 techniques
        tech_data = df[df['Technique'] == technique]
        if not tech_data.empty:
            strengths = []
            weaknesses = []
            
            for metric in ['Faithfulness', 'Relevance', 'Completeness', 'Clarity']:
                if metric in tech_data.columns:
                    score = tech_data[metric].mean()
                    if score > 0.7:
                        strengths.append(f"{metric} ({score:.3f})")
                    elif score < 0.5:
                        weaknesses.append(f"{metric} ({score:.3f})")
            
            print(f"\n   {technique}:")
            if strengths:
                print(f"     Strengths: {', '.join(strengths)}")
            if weaknesses:
                print(f"     Areas for improvement: {', '.join(weaknesses)}")

def main():
    """Main function to run visualization and analysis"""
    print("📊 RAG Evaluation Results Visualization")
    print("="*50)
    
    # Load results
    results = load_latest_results()
    if not results:
        return
    
    # Create visualizations
    try:
        df, technique_scores = create_comprehensive_visualizations(results)
        
        # Print detailed analysis
        print_detailed_analysis(df, technique_scores)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Visualization saved as 'rag_evaluation_comprehensive.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
