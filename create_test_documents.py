"""
Create diverse test documents for RAG evaluation
"""

import os
import json
import csv
from pathlib import Path

def create_test_documents():
    """Create diverse test documents in different formats"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Technical Documentation (TXT)
    tech_doc = """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

## Key Concepts:
- Supervised Learning: Learning with labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data  
- Reinforcement Learning: Learning through trial and error

## Common Algorithms:
1. Linear Regression - for predicting continuous values
2. Decision Trees - for classification and regression
3. Neural Networks - for complex pattern recognition
4. K-Means Clustering - for grouping similar data points

## Applications:
- Image Recognition: Identifying objects in photos
- Natural Language Processing: Understanding human language
- Recommendation Systems: Suggesting products or content
- Fraud Detection: Identifying suspicious transactions

## Performance Metrics:
- Accuracy: Overall correctness of predictions
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall

## Best Practices:
1. Data Quality: Ensure clean, representative datasets
2. Feature Engineering: Create meaningful input variables
3. Model Selection: Choose appropriate algorithms for the problem
4. Cross-Validation: Test models on unseen data
5. Hyperparameter Tuning: Optimize model parameters
"""
    
    with open(data_dir / "machine_learning_guide.txt", "w") as f:
        f.write(tech_doc)
    
    # 2. Sales Data (CSV)
    sales_data = [
        ["Product", "Quarter", "Sales", "Region", "Manager", "Cost", "Profit"],
        ["Laptop", "Q1", "150000", "North", "Alice", "120000", "30000"],
        ["Laptop", "Q2", "180000", "North", "Alice", "140000", "40000"],
        ["Tablet", "Q1", "95000", "South", "Bob", "70000", "25000"],
        ["Tablet", "Q2", "120000", "South", "Bob", "85000", "35000"],
        ["Phone", "Q1", "250000", "East", "Carol", "180000", "70000"],
        ["Phone", "Q2", "280000", "East", "Carol", "200000", "80000"],
        ["Desktop", "Q1", "75000", "West", "David", "55000", "20000"],
        ["Desktop", "Q2", "85000", "West", "David", "60000", "25000"],
        ["Monitor", "Q1", "45000", "North", "Alice", "30000", "15000"],
        ["Monitor", "Q2", "52000", "North", "Alice", "35000", "17000"],
    ]
    
    with open(data_dir / "sales_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sales_data)
    
    # 3. Configuration Data (JSON)
    config_data = {
        "application": {
            "name": "DataProcessor",
            "version": "2.1.0",
            "environment": "production",
            "features": {
                "authentication": True,
                "logging": True,
                "caching": False,
                "monitoring": True
            }
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "analytics_db",
            "connection_pool": {
                "min_connections": 5,
                "max_connections": 20,
                "timeout": 30
            },
            "backup": {
                "enabled": True,
                "frequency": "daily",
                "retention_days": 30
            }
        },
        "api": {
            "rate_limiting": {
                "requests_per_minute": 1000,
                "burst_limit": 50
            },
            "authentication": {
                "method": "JWT",
                "token_expiry": "24h",
                "refresh_enabled": True
            },
            "endpoints": [
                "/api/v1/users",
                "/api/v1/data",
                "/api/v1/analytics",
                "/api/v1/reports"
            ]
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "retention_days": 30,
            "destinations": ["file", "console", "syslog"]
        }
    }
    
    with open(data_dir / "app_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # 4. Business Process (TXT)
    business_doc = """
# Customer Onboarding Process

## Overview
The customer onboarding process is designed to ensure new customers have a smooth transition into using our services. This comprehensive process typically takes 60 days and involves multiple departments.

## Steps:

### 1. Initial Contact (Day 1)
- Welcome email sent automatically within 1 hour of signup
- Account setup instructions provided via email and SMS
- Dedicated customer success manager assigned based on account size
- Initial consultation call scheduled within 24 hours

### 2. Account Setup (Days 1-3)
- Customer completes profile information in onboarding portal
- Payment method verification and billing setup
- Service preferences configuration and customization
- Security settings establishment including 2FA
- Integration requirements assessment

### 3. Training Phase (Days 4-7)
- Personalized training session scheduled based on customer needs
- Access to online tutorials and comprehensive documentation
- Practice environment setup with sample data
- Q&A session with support team and technical specialists
- Custom training materials provided for complex integrations

### 4. Go-Live (Day 8)
- Production environment access granted after security review
- Initial usage monitoring and performance baseline establishment
- 24/7 support availability for first 30 days
- Performance baseline establishment and SLA activation
- Backup and disaster recovery procedures testing

### 5. Follow-up (Days 15, 30, 60)
- Check-in calls with customer success manager
- Usage analytics review and optimization recommendations
- Feature adoption assessment and training refreshers
- Feedback collection and analysis for process improvement
- Account growth planning and expansion discussions

## Success Metrics:
- Time to first value: < 7 days (target: 5 days)
- Feature adoption rate: > 80% within 30 days
- Customer satisfaction score: > 4.5/5 throughout process
- Support ticket volume: < 2 per month after go-live
- Renewal probability: > 95% for properly onboarded customers

## Department Responsibilities:
- Sales: Initial handoff and relationship transfer
- Customer Success: Primary ownership and coordination
- Technical Support: Training and troubleshooting
- Security: Access provisioning and compliance
- Billing: Payment setup and subscription management
"""
    
    with open(data_dir / "onboarding_process.txt", "w") as f:
        f.write(business_doc)
    
    # 5. Financial Report (TXT)
    financial_doc = """
# Q2 2025 Financial Report

## Executive Summary
The company demonstrated strong financial performance in Q2 2025, with revenue growth of 15% year-over-year and improved operational efficiency across all business units.

## Revenue Analysis
- Total Revenue: $12.5 million (up 15% from Q2 2024)
- Recurring Revenue: $9.8 million (78% of total revenue)
- New Customer Revenue: $2.7 million (22% of total revenue)

### Revenue by Product Line:
1. Software Licenses: $7.2 million (58% of total)
2. Professional Services: $3.1 million (25% of total)
3. Support & Maintenance: $2.2 million (17% of total)

### Revenue by Geography:
- North America: $8.75 million (70%)
- Europe: $2.5 million (20%)
- Asia-Pacific: $1.25 million (10%)

## Cost Structure
- Cost of Goods Sold: $4.2 million (34% of revenue)
- Operating Expenses: $6.8 million (54% of revenue)
- EBITDA: $1.5 million (12% of revenue)

### Major Expense Categories:
- Personnel Costs: $5.2 million (42% of revenue)
- Technology Infrastructure: $1.8 million (14% of revenue)
- Sales & Marketing: $2.1 million (17% of revenue)
- General & Administrative: $1.5 million (12% of revenue)

## Key Performance Indicators
- Customer Acquisition Cost (CAC): $2,500 (down 10% from Q1)
- Customer Lifetime Value (CLV): $45,000 (up 8% from Q1)
- Monthly Recurring Revenue (MRR): $3.27 million
- Churn Rate: 2.1% (industry benchmark: 3.5%)
- Net Promoter Score: 67 (industry benchmark: 50)

## Future Outlook
Based on current trends and pipeline analysis, we project:
- Q3 2025 Revenue: $13.2-13.8 million
- Q4 2025 Revenue: $14.1-14.9 million
- Annual Growth Rate: 18-22%
"""
    
    with open(data_dir / "financial_report.txt", "w") as f:
        f.write(financial_doc)
    
    print("✅ Test documents created successfully!")
    print(f"📁 Documents saved in: {data_dir.absolute()}")
    print("\nCreated documents:")
    print("- machine_learning_guide.txt (Technical documentation)")
    print("- sales_data.csv (Structured data)")
    print("- app_config.json (Configuration data)")
    print("- onboarding_process.txt (Business process)")
    print("- financial_report.txt (Financial data)")

if __name__ == "__main__":
    create_test_documents()
