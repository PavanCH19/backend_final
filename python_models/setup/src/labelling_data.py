import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_domain_requirements(filepath='data/domain_requirements.json'):
    """Load domain requirements from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_resumes(filepath='data/synthetic_resumes.json'):
    """Load resumes from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_resume_metrics(resume, required_skills):
    """Calculate metrics for a single resume"""
    candidate_skills = set(resume["skills"])
    skill_match_ratio = len(required_skills.intersection(candidate_skills)) / len(required_skills)
    test_score_norm = resume["test_score"] / 100.0
    project_count = len(resume["projects"])
    
    return skill_match_ratio, test_score_norm, project_count


def determine_label(skill_match_ratio, test_score_norm, project_count):
    """Determine label based on metrics using defined rules"""
    if skill_match_ratio >= 0.70 and test_score_norm >= 0.75 and project_count >= 1:
        return "Fit"
    elif (0.40 <= skill_match_ratio < 0.70) or (0.50 <= test_score_norm < 0.75):
        return "Partial Fit"
    else:
        return "Not Fit"


def calculate_labels(resumes, domain_requirements):
    """Calculate labels for resumes based on rules"""
    labeled_resumes = []
    
    for resume in resumes:
        # Find the domain key for this resume
        preferred_domain_key = next(
            (k for k, v in domain_requirements.items() if v["domain"] == resume["preferred_domain"]), 
            None
        )
        if not preferred_domain_key:
            continue
            
        required_skills = set(domain_requirements[preferred_domain_key]["required_skills"])
        
        # Calculate metrics
        skill_match_ratio, test_score_norm, project_count = calculate_resume_metrics(resume, required_skills)
        
        # Determine label
        label = determine_label(skill_match_ratio, test_score_norm, project_count)
        
        # Create labeled resume
        resume_with_label = resume.copy()
        resume_with_label.update({
            "skill_match_ratio": skill_match_ratio,
            "test_score_norm": test_score_norm,
            "project_count": project_count,
            "label": label
        })
        labeled_resumes.append(resume_with_label)
    
    return labeled_resumes


def save_labeled_resumes(labeled_resumes, filepath='data/labeled_synthetic_resumes.json'):
    """Save labeled resumes to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(labeled_resumes, f, indent=2)
    print(f"Generated {len(labeled_resumes)} resumes → {filepath}\n")


def print_dataset_statistics(df):
    """Print basic dataset statistics"""
    label_counts = df['label'].value_counts()
    
    print("Dataset Statistics:")
    print(f"Total: {len(df)} | Avg Test Score: {df['test_score'].mean():.1f}")
    print(f"\nLabel Distribution:\n{label_counts}\n")
    
    return label_counts


def print_domain_analysis(df):
    """Print detailed analysis by domain"""
    print("\n=== Analysis by Domain ===")
    domain_label_crosstab = pd.crosstab(df['preferred_domain'], df['label'])
    print(domain_label_crosstab)
    return domain_label_crosstab


def plot_overall_distribution(ax, label_counts, colors):
    """Plot overall label distribution pie chart"""
    ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
    ax.set_title('Overall Label Distribution')


def plot_domain_distribution(ax, domain_label_crosstab, colors):
    """Plot label distribution by domain"""
    domain_label_crosstab.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Label Distribution by Domain')
    ax.set_xlabel('Domain')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Label')


def plot_skill_match_histogram(ax, df):
    """Plot skill match ratio histogram by label"""
    for label in df['label'].unique():
        data = df[df['label'] == label]['skill_match_ratio']
        ax.hist(data, alpha=0.6, label=label, bins=20)
    ax.set_xlabel('Skill Match Ratio')
    ax.set_title('Skill Match Ratio by Label')
    ax.legend()


def plot_test_score_histogram(ax, df):
    """Plot test score distribution by label"""
    for label in df['label'].unique():
        data = df[df['label'] == label]['test_score']
        ax.hist(data, alpha=0.6, label=label, bins=20)
    ax.set_xlabel('Test Score')
    ax.set_title('Test Score Distribution by Label')
    ax.legend()


def plot_skill_vs_test_scatter(ax, df, colors):
    """Plot scatter plot of skill match vs test score"""
    for i, label in enumerate(df['label'].unique()):
        data = df[df['label'] == label]
        ax.scatter(data['skill_match_ratio'], data['test_score_norm'], 
                  alpha=0.6, label=label, color=colors[i])
    ax.set_xlabel('Skill Match Ratio')
    ax.set_ylabel('Test Score (Normalized)')
    ax.set_title('Skill Match vs Test Score')
    ax.legend()


def plot_project_count_distribution(ax, df):
    """Plot project count by label"""
    project_counts_by_label = df.groupby(['label', 'project_count']).size().unstack(fill_value=0)
    project_counts_by_label.plot(kind='bar', ax=ax)
    ax.set_title('Project Count by Label')
    ax.set_xlabel('Label')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Projects')


def create_visualization(df, label_counts, domain_label_crosstab, save_path='data/labeling_analysis.png'):
    """Create comprehensive visualization with all plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['lightgreen', 'orange', 'lightcoral']
    
    # Create all plots
    plot_overall_distribution(axes[0, 0], label_counts, colors)
    plot_domain_distribution(axes[0, 1], domain_label_crosstab, colors)
    plot_skill_match_histogram(axes[0, 2], df)
    plot_test_score_histogram(axes[1, 0], df)
    plot_skill_vs_test_scatter(axes[1, 1], df, colors)
    plot_project_count_distribution(axes[1, 2], df)
    
    # Save and display
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Analysis saved → {save_path}")


def analyze_labeled_resumes(labeled_resumes):
    """Perform complete analysis on labeled resumes"""
    # Convert to DataFrame
    df = pd.DataFrame(labeled_resumes)
    
    # Print statistics
    label_counts = print_dataset_statistics(df)
    domain_label_crosstab = print_domain_analysis(df)
    
    # Create visualizations
    create_visualization(df, label_counts, domain_label_crosstab)
    
    return df


# def main():
#     """Main function to execute resume labeling and analysis"""
#     # Load data
#     domain_requirements = load_domain_requirements()
#     synthetic_resumes = load_resumes()
    
#     # Calculate labels
#     labeled_resumes = calculate_labels(synthetic_resumes, domain_requirements)
    
#     # Save labeled resumes
#     save_labeled_resumes(labeled_resumes)
    
#     # Analyze and visualize
#     analyze_labeled_resumes(labeled_resumes)


# if __name__ == "__main__":
#     main()