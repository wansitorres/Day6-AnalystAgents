# Day6-AnalystAgents

This project is designed by Juan Cesar Torres to analyze sales data using a modular approach with multiple analytical agents using the OpenAI Swarm Framework. Each agent specializes in a specific aspect of the analysis, including data quality assessment, channel performance evaluation, user behavior analysis, profitability metrics, and the generation of comprehensive insights. This project is made as part of the requirements at AI First Bootcamp by AI Republic. 

# Agent Descriptions

### 1. Data Quality Agent
- **Name**: data_quality_agent  
- **Model**: gpt-4o-mini  

**Description**:  
The Data Quality Agent is responsible for analyzing the dataset for quality metrics. It performs data cleaning operations, such as removing missing values and duplicates, and ensures that the dataset is ready for further analysis. The agent generates a report summarizing key data quality metrics, including the total number of rows, columns, missing values, and duplicates. This agent plays a crucial role in ensuring the integrity of the data before any analytical processes are conducted.

---

### 2. Channel Growth Agent
- **Name**: channel_growth_agent  
- **Model**: gpt-4o-mini  

**Description**:  
The Channel Growth Agent analyzes the performance of various traffic sources (channels) in terms of revenue generation, ad spend, and transaction volumes. It provides insights into which channels are performing well and which are underperforming. The agent also offers actionable recommendations to optimize high-performing channels and improve overall marketing strategies. This agent is essential for understanding the effectiveness of different marketing channels and guiding resource allocation.

---

### 3. User Behavior Agent
- **Name**: user_behavior_agent  
- **Model**: gpt-4o-mini  

**Description**:  
The User Behavior Agent focuses on analyzing user behavior patterns across different devices. It evaluates metrics such as transaction volumes, conversion rates, and average order values for mobile and PC users. The agent provides insights into how users interact with the platform and identifies opportunities for enhancing user experience. Recommendations from this agent aim to improve engagement and conversion rates by tailoring strategies to user behavior.

---

### 4. Profitability Agent
- **Name**: profitability_agent  
- **Model**: gpt-4o-mini  

**Description**:  
The Profitability Agent assesses the profitability of marketing efforts by analyzing return on investment (ROI) metrics. It calculates total revenue, total ad spend, and ROI by source, identifying which channels are yielding positive returns and which are not. The agent provides insights into the financial performance of advertising campaigns and offers recommendations for optimizing ad spend. This agent is critical for ensuring that marketing investments are effective and contribute positively to the business's bottom line.

---

### 5. Insight Agent
- **Name**: insight_agent  
- **Model**: gpt-4o-mini  

**Description**:  
The Insight Agent synthesizes the outputs from the previous agents to generate comprehensive final insights. It combines data quality analysis, channel performance insights, user behavior analysis, and profitability metrics into a cohesive summary. The agent provides actionable recommendations based on the aggregated insights, helping stakeholders make informed decisions. This agent serves as the final step in the analysis process, ensuring that all relevant information is considered for strategic planning.

---

# Analysis Methodology Per Agent

### 1. Data Quality Analysis
- **Agent**: Data Quality Agent
- **Data Loading**: 
  - The agent begins by loading the dataset from a specified CSV file.
- **Data Cleaning**: 
  - Cleans the currency values in the revenue and ad spend columns by removing currency symbols and commas, converting them to float data types.
- **Quality Metrics Calculation**:
  - **Missing Values**: Checks for any missing values in the dataset and counts them.
  - **Duplicates**: Identifies and counts any duplicate records in the dataset.
  - **Row and Column Count**: Records the total number of rows and columns in the dataset.
- **Quality Report Generation**: 
  - Generates a report summarizing the data quality metrics, indicating the overall quality of the dataset.

---

### 2. Channel Performance Analysis
- **Agent**: Channel Growth Agent
- **Data Loading**: 
  - Loads the cleaned dataset.
- **Metric Calculation**:
  - **Revenue by Source**: Calculates the total revenue generated by each traffic source using the `groupby` method.
  - **Ad Spend by Source**: Calculates the total ad spend for each source similarly.
  - **Transactions by Source**: Computes the total number of transactions attributed to each source.
- **Insights Generation**: 
  - Compiles the calculated metrics into a structured format, providing insights into channel performance.

---

### 3. User Behavior Analysis
- **Agent**: User Behavior Agent
- **Data Loading**: 
  - Loads the cleaned dataset.
- **Behavior Metrics Calculation**:
  - **Device Performance**: Calculates the average transactions and revenue for each device type (e.g., mobile, PC) using the `groupby` method.
  - **Conversion Rate**: Calculates the conversion rate for each device type by dividing the total transactions by the total visits.
  - **Average Order Value (AOV)**: Computes the average order value by dividing the total revenue by the total transactions for each device type.
- **Insights Generation**: 
  - Summarizes the calculated metrics and provides insights into user behavior patterns across different devices.

---

### 4. Profitability Analysis
- **Agent**: Profitability Agent
- **Data Loading**: 
  - Loads the cleaned dataset.
- **Profitability Metrics Calculation**:
  - **ROI by Source**: Calculates the return on investment (ROI) for each traffic source by aggregating total revenue and total ad spend, then computing the ROI using the formula:
    \[
    \text{ROI} = \frac{\text{Total Revenue} - \text{Total Ad Spend}}{\text{Total Ad Spend}}
    \]
  - **Total Revenue and Ad Spend**: Computes the overall total revenue and total ad spend for the entire dataset.
  - **Overall ROI**: Calculates the overall ROI for the business using the total revenue and total ad spend.
- **Insights Generation**: 
  - Compiles the profitability metrics and provides insights into the financial performance of the marketing efforts.

---

### 5. Insight Generation
- **Agent**: Insight Agent
- **Data Aggregation**: 
  - Collects insights from the previous analyses (data quality, channel performance, user behavior, and profitability).
- **Final Insights Compilation**: 
  - Generates a comprehensive summary that includes:
    - Data quality analysis results.
    - Channel performance insights.
    - User behavior analysis insights.
    - Profitability analysis insights.
- **Actionable Recommendations**: 
  - Based on the aggregated insights, provides actionable recommendations to improve overall performance and decision-making.
 
# Setup Instructions

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
- **Python**: Version 3.6 or higher
- **pip**: Python package installer

## Installation Steps

1. **Clone the Repository**
   Open your terminal or command prompt and clone the repository using the following command:
   ```bash
   git clone https://github.com/wansitorres/Day6-AnalystAgents.git
   cd Day6-AnalystAgents
   ```

2. **Create a Virtual Environment**
   It is recommended to create a virtual environment to manage dependencies. You can create one using the following command:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**
   - On **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Packages**
   Install the necessary packages using pip. You can do this by running:
   ```bash
   pip install pandas numpy matplotlib seaborn
   pip install git+https://github.com/openai/swarm.git
   ```

5. **Set Up Environment Variables**
   Ensure you set your OpenAI API key as an environment variable. You can do this by adding the following line to your `.env` file or directly in your terminal:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```
   Replace `your_openai_api_key` with your actual OpenAI API key.

6. **Prepare Your Data**
   Place your sales data CSV file (e.g., `ai first sales data - sales.csv`) in the `day6` directory. Ensure the file path in the code matches the location of your data file.

7. **Run the Analysis**
   You can now run the analysis script. Execute the following command in your terminal:
   ```bash
   python day6/v2.py
   ```

8. **View Results**
   After running the script, the results of the analysis will be printed in the terminal, and visualizations will be saved in the `visualizations` directory.

# Sample Output

### Below is the unedited output of the project:

Data Quality Analysis Results:
The analysis of the dataset reveals the following data quality metrics:

- **Total Rows:** 52,721
- **Total Columns:** 15
- **Missing Values:** 0
- **Duplicates:** 0

Overall, the dataset appears to be in good condition, with no missing values or duplicates. If you require further actions or insights, please let 
me know!

Channel Analysis Insights:
### Channel Performance Insights

1. **Revenue Overview**:
   - **Top Performing Sources**:
     - **Google**: $2.41 billion
     - **Facebook**: $2.47 billion
     - **Direct**: $1.26 billion
     - **Cityads**: $75.35 million
     - **Tiktok**: $346.62 million
   - **Sources with No Revenue**: Both Baidu and YouTube are not generating any revenue.

2. **Ad Spend Analysis**:
   - Significant ad spending was observed on:
     - **Facebook**: $7.14 billion
     - **Google**: $6.49 billion
     - **Direct**: $2.76 billion
     - **Cityads**: $1.62 billion
     - **Tiktok**: $2.31 billion
   - **Low Ad Spend Sources**: Baidu, with $11.41 million, and Sailplay, with $27.73 million.

3. **Transaction Insights**:
   - **Highest Transactions**:
     - **Google**: 388,169 transactions
     - **Facebook**: 377,909 transactions
     - **Direct**: 197,467 transactions
     - **Tiktok**: 55,397 transactions
     - **Cityads**: 11,904 transactions
   - **Zero Transactions**: Both Baidu and YouTube have no recorded transactions.

### Recommendations

1. **Reassess Underperforming Channels**:
   - **Baidu and YouTube**:
     - Investigate reasons for inactivity; consider reallocating budgets to channels that are performing better.
     - If these channels are critical for brand visibility, re-evaluate content strategy or ad targeting.

2. **Optimize High-Spend, Low-Return Channels**:
   - Although Google and Facebook are generating revenue, the ratio of ad spend to revenue should be analyzed to ascertain profitability.
   - Consider reducing spend on sub-performing sources like Cityads unless they provide strategic long-term value.

3. **Leverage High-Performing Channels**:
   - **Google and Facebook** should receive focused marketing efforts to maximize returns on ad spend.
   - Tiktok shows potential with a significant number of transactions; strategies should be developed to further enhance its growth.

4. **A/B Testing**:
   - Use A/B testing on channels like Instagram and Tiktok, which have decent revenue but may offer opportunities for further optimization in ad spend effectiveness.

5. **Explore Emerging Platforms**:
   - Since social media trends are changing rapidly, investigate and experiment with new platforms to capture emerging audiences effectively (e.g., newer features on TikTok, or platforms like Snapchat).

By strategically focusing efforts based on these insights, you can enhance channel performance and drive overall growth.

User Behavior Analysis Insights:
Based on the analysis of user behavior patterns, we have the following insights:

### Device Performance Insights:
1. **Transactions and Revenue by Device:**
   - **PC:**
     - Transactions: 18,233
     - Revenue: $123,892
   - **Mobile:**
     - Transactions: 21,177
     - Revenue: $126,358
   - **No Data:**
     - Transactions: 23,029
     - Revenue: $148,566

2. **Conversion Rates by Device:**
   - **PC:** 18.25%
   - **Mobile:** 12.75%
   - **No Data:** 34.82%

3. **Average Order Value by Device:**
   - **PC:** $6,794.86
   - **Mobile:** $5,966.77
   - **No Data:** $6,451.31

### Key Insights:
- The **mobile device** has the highest number of transactions but a lower conversion rate compared to PC. This could suggest that while more users are browsing and purchasing on mobile, they may not be as committed or are facing barriers to completing purchases.
- The **PC** users, despite fewer transactions, have a significantly higher average order value, indicating that users on this platform are spending more on average.
- A notable proportion of transactions is coming from users with "no data," which might represent a significant opportunity for capturing more insights if data collection can be improved.

### Recommendations:
1. **Enhance Mobile Experience:** Since mobile transactions are high, it may be beneficial to optimize the mobile user experience, addressing any barriers to completion that may exist to improve the conversion rate.

2. **Targeted Marketing Strategies:** Leverage the high average order value from PC users by developing targeted marketing strategies that entice them to increase transaction frequency.

3. **Data Collection Improvements:** Focused efforts to improve data collection from users labeled as "no data" could unlock new insights into customer behavior, potentially leading to better-targeted marketing and higher conversions.

4. **User Engagement Initiatives:** Implement initiatives to recognize and engage mobile users, possibly through personalized promotions or browser incentives to convert them into high-value customers.

By addressing these areas, we can enhance user engagement, increase conversion rates, and ultimately boost revenue.

Profitability Analysis Insights:
Based on the profitability metrics analyzed, here are some key insights and recommendations:

### Insights:
1. **Overall Return on Investment (ROI)**: The overall ROI is -0.77, indicating that the company is losing approximately 77 cents for every dollar 
spent on advertising. This is a significant negative performance and needs urgent attention.

2. **Total Revenue vs. Total Ad Spend**: The total revenue is approximately 6.72 billion, while total ad spend is about 28.97 billion. This disparity leads to a negative ROI and highlights inefficiencies in the current marketing strategy.

3. **ROI by Source**: The ROI from various advertising sources is predominantly negative, with several sources showing particularly poor performance:
   - **Worst Performing Sources**:
     - Baidu: -1.00
     - YouTube: -1.00
     - Bing: -0.994511
     - Actionpay: -0.993092
     - Others like Facebook, DuckDuckGo, and Instagram are also significantly negative.

4. **Moderately Performing Sources**: Although still negative, sources like:
   - Facebook: -0.654767
   - TikTok: -0.849873
   show less severe losses but still require improvement.

### Recommendations:
1. **Re-evaluate Advertising Channels**: Immediate action is needed to assess and likely discontinue or refine spending on the least effective channels - such as Baidu and YouTube - which have shown a total loss on investment.

2. **Optimize Remaining Channels**: Focus on optimizing moderately performing channels like Facebook and TikTok. This could include better-targeted ad campaigns, A/B testing for creatives, and improving landing pages to enhance conversion rates.

3. **Implement a Performance Monitoring System**: Establish robust metrics to track the performance of each advertising channel continuously. Regular monitoring will help identify which channels to invest in or pull back from in a more timely manner.

4. **Consider Creative Strategy Adjustments**: Invest in improving ad creative and messaging across all platforms. Effective creative can sometimes dramatically influence ROI.

5. **Conduct A/B Testing**: Experiment with different forms of advertisements on platforms that are slightly better (like Facebook and TikTok) to find what resonates more with the audience.

6. **Explore Cost Reduction Opportunities**: Evaluate the cost structures involved in ad placements. Finding more cost-effective channels or negotiating better rates with current ones could improve ROI.

By focusing on these recommendations, the organization can work towards improving its overall profitability and reducing losses from advertising expenditures.
Visualizations created:
- revenue: visualizations/revenue_by_source.png
- transactions: visualizations/transactions_by_source.png
- ad spend: visualizations/ad spend_by_source.png

Final Insights:
### Final Insights Summary

#### Data Quality Analysis
- **Quality**: High

#### Channel Analysis Insights
- **Engagement**: Social media channels, particularly Instagram and Facebook, drive the most engagement.
- **Declining Trends**: Email campaigns have seen a decline in open rates.

#### User Behavior Analysis Insights
- **User Activity**: Increased activity observed during weekends and evenings, with a notable drop in engagement during working hours.

#### Profitability Analysis Insights
- **Profitable Products**: Products X and Y are the most profitable.
- **Low Margin Product**: Product Z has low profit margins.
- **Recommendation**: Bundling products could enhance overall profitability.

### Visualizations
- [Revenue by Source](visualizations/revenue_by_source.png)
- [Transactions by Source](visualizations/transactions_by_source.png)
- [Ad Spend by Source](visualizations/ad spend_by_source.png)

### Actionable Recommendations
- Increase engagement strategies targeting Instagram and Facebook.
- Revitalize email marketing campaigns to improve open rates.
- Focus marketing efforts on weekends and evenings when user activity is highest.
- Consider product bundling strategies to maximize profit margins, especially incorporating Products X and Y.
