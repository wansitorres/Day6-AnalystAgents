import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from swarm import Agent, Swarm
import os

os.environ["OPENAI_API_KEY"] = ""

def analyze_data_quality():
    """Analyze data quality metrics and clean the dataset."""
    data = pd.read_csv(r"E:\Downloads\AIRepublicDay3\day6\ai first sales data - sales.csv")
    
    # Clean currency values
    data['revenue'] = data['revenue'].str.replace('₱', '').str.replace('â‚±', '').str.replace(',', '').astype(float)
    data['ad spend'] = data['ad spend'].str.replace('₱', '').str.replace('â‚±', '').str.replace(',', '').astype(float)
    
    # Save cleaned data
    data.to_csv(r"E:\Downloads\AIRepublicDay3\day6\cleaned.csv", index=False)
    
    # Analyze quality
    quality_report = {
        'missing_values': data.isnull().sum().sum(),
        'duplicates': data.duplicated().sum(),
        'total_rows': len(data),
        'total_columns': len(data.columns)
    }
    
    return quality_report

def analyze_channels():
    """Analyze performance metrics for traffic sources."""
    data = pd.read_csv(r"E:\Downloads\AIRepublicDay3\day6\cleaned.csv")
    
    # Calculate channel metrics
    channel_metrics = {
        'revenue_by_source': data.groupby('source')['revenue'].sum(),
        'ad_spend_by_source': data.groupby('source')['ad spend'].sum(),
        'transactions_by_source': data.groupby('source')['transactions'].sum()
    }
    
    return channel_metrics

def analyze_user_behavior():
    """Analyze user behavior patterns."""
    data = pd.read_csv(r"E:\Downloads\AIRepublicDay3\day6\cleaned.csv")
    
    # Calculate behavior metrics
    behavior_metrics = {
        'device_performance': data.groupby('device_type')[['transactions', 'revenue']].mean(),
        'conversion_rate': data.groupby('device_type')['transactions'].sum() / data.groupby('device_type')['visits'].sum(),
        'avg_order_value': data.groupby('device_type')['revenue'].sum() / data.groupby('device_type')['transactions'].sum()
    }
    
    return behavior_metrics

def analyze_profitability():
    """Analyze profitability metrics."""
    data = pd.read_csv(r"E:\Downloads\AIRepublicDay3\day6\cleaned.csv")
    
    # Calculate profitability metrics
    profitability_metrics = {
        'roi_by_source': (data.groupby('source').agg(
            total_revenue=('revenue', 'sum'),
            total_ad_spend=('ad spend', 'sum')
        )).assign(
            roi=lambda x: (x['total_revenue'] - x['total_ad_spend']) / x['total_ad_spend']
        )['roi'],
        'total_revenue': data['revenue'].sum(),
        'total_ad_spend': data['ad spend'].sum(),
        'overall_roi': (data['revenue'].sum() - data['ad spend'].sum()) / data['ad spend'].sum()
    }
    
    return profitability_metrics

def create_visualizations():
    """Create and save visualizations."""
    data = pd.read_csv(r"E:\Downloads\AIRepublicDay3\day6\cleaned.csv")
    
    # Create visualizations directory
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Create and save visualizations
    visualization_paths = {}
    metrics = ["revenue", "transactions", "ad spend"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x='source', y=metric)
        plt.title(f"{metric} by Source")
        plt.xticks(rotation=45)
        path = f"visualizations/{metric}_by_source.png"
        plt.savefig(path)
        plt.close()
        visualization_paths[metric] = path
    
    print("Visualizations created:")
    for metric, path in visualization_paths.items():
        print(f"- {metric}: {path}")
    
    return visualization_paths

def generate_final_insights(data_quality, channel_insights, user_behavior_insights, profitability_insights):
    """Generate comprehensive final insights based on previous analyses."""
    final_insights = f"""
### Final Insights Summary

#### Data Quality Analysis
{data_quality}

#### Channel Analysis Insights
{channel_insights}

#### User Behavior Analysis Insights
{user_behavior_insights}

#### Profitability Analysis Insights
{profitability_insights}

"""
    return final_insights

# Define the agents
data_quality_agent = Agent(
    name="Data Quality Agent",
    model="gpt-4o-mini",
    instructions="""You are a Data Quality Analyst. Here are your instructions:
    1. Analyze and clean the data.
    2. After cleaning the data, provide a report on data quality metrics.""",
    functions=[analyze_data_quality]
)

channel_growth_agent = Agent(
    name="Channel Growth Agent",
    model="gpt-4o-mini",
    instructions="""You are a Channel Growth Analyst. Here are your instructions:
    1. Provide insights and recommendations based on channel performance.
    2. Do not transfer to the next agent; just provide your insights.""",
    functions=[analyze_channels]
)

user_behavior_agent = Agent(
    name="User Behavior Agent",
    model="gpt-4o-mini",
    instructions="""You are a User Behavior Analyst. Here are your instructions:
    1. Provide insights and recommendations based on user behavior patterns.
    2. Do not transfer to the next agent; just provide your insights.""",
    functions=[analyze_user_behavior]
)

profitability_agent = Agent(
    name="Profitability Agent",
    model="gpt-4o-mini",
    instructions="""You are a Profitability Analyst. Here are your instructions: 
    1. Provide insights and recommendations based on profitability metrics.
    2. Do not transfer to the next agent; just provide your insights.""",
    functions=[analyze_profitability]
)

insight_agent = Agent(
    name="Insight Agent",
    model="gpt-4o-mini",
    instructions="""You are an Insight Delivery Specialist. Here are your instructions:
    1. Provide insights about the data provided and summarize the insights from other agents, including actionable recommendations.
    2. Be concise.
    3. Create the visualizations.""",
    functions=[create_visualizations, generate_final_insights]
)

if __name__ == "__main__":
    client = Swarm()
    
    # Run the Data Quality Agent
    data_quality_response = client.run(
        agent=data_quality_agent,
        messages=[{
            "role": "user", 
            "content": "Please analyze my dataset for data quality."
        }]
    )
    
    print("\nData Quality Analysis Results:")
    print(data_quality_response.messages[-1]["content"])  # Print insights from Data Quality Agent
    
    # Run the Channel Growth Agent
    channel_response = client.run(
        agent=channel_growth_agent,
        messages=[{
            "role": "user",
            "content": "Analyze channel performance with the cleaned data."
        }]
    )
    
    print("\nChannel Analysis Insights:")
    print(channel_response.messages[-1]["content"])  # Print insights from Channel Growth Agent
    
    # Run the User Behavior Agent
    user_behavior_response = client.run(
        agent=user_behavior_agent,
        messages=[{
            "role": "user",
            "content": "Analyze user behavior patterns."
        }]
    )
    
    print("\nUser Behavior Analysis Insights:")
    print(user_behavior_response.messages[-1]["content"])  # Print insights from User Behavior Agent
    
    # Run the Profitability Agent
    profitability_response = client.run(
        agent=profitability_agent,
        messages=[{
            "role": "user",
            "content": "Analyze profitability metrics."
        }]
    )
    
    print("\nProfitability Analysis Insights:")
    print(profitability_response.messages[-1]["content"])  # Print insights from Profitability Agent
    
    # Run the Insight Agent
    insight_response = client.run(
        agent=insight_agent,
        messages=[{
            "role": "user",
            "content": "Generate comprehensive insights from all collected metrics."
        }]
    )
    
    print("\nFinal Insights:")
    print(insight_response.messages[-1]["content"])  # Print insights from Insight Agent
    
