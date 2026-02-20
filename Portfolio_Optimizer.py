import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📈 Smart Portfolio Optimizer")
st.markdown("### Optimize your investment portfolio using Modern Portfolio Theory")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Investment amount
    investment = st.number_input(
        "💰 Investment Amount (₹)",
        min_value=1000.0,
        max_value=10000000000.0,
        value=100000.0,
        step=10000.0,
        format="%.2f"
    )
    
    # Risk profile
    risk_profile = st.selectbox(
        "🎯 Risk Appetite",
        options=["Low", "Medium", "High"],
        index=1,
        help="Low: Minimum volatility | Medium: Best risk-return balance | High: Maximum returns"
    )
    
    # Time period
    time_period = st.slider(
        "📅 Investment Horizon (years)",
        min_value=1,
        max_value=10,
        value=5,
        help="Historical data period for analysis"
    )
    
    st.divider()
    
    # Stock selection
    st.subheader("📊 Stock Universe")
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]
    
    use_custom = st.checkbox("Customize stock selection")
    
    if use_custom:
        tickers_input = st.text_area(
            "Enter ticker symbols (comma-separated)",
            value=", ".join(default_tickers),
            height=100
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        tickers = default_tickers
    
    st.caption(f"Selected: {len(tickers)} stocks")
    
    st.divider()
    
    # Optimize button
    optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

# Main content area
if optimize_button:
    with st.spinner("🔄 Downloading market data and optimizing..."):
        try:
            # Download data
            prices = yf.download(
                tickers,
                period=f"{time_period}y",
                interval="1d",
                auto_adjust=False,
                progress=False
            )
            
            if prices.empty:
                st.error("❌ Failed to download data. Please check ticker symbols.")
                st.stop()
            
            # Extract adjusted close prices
            adj_close = prices["Adj Close"]
            
            # Calculate returns
            returns = adj_close.pct_change().dropna()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if returns.empty:
                st.error("❌ No valid return data available.")
                st.stop()
            
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Portfolio optimization functions
            def portfolio_performance(weights, mean_returns, cov_matrix):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_return, portfolio_volatility
            
            def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04/252):
                portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe_ratio
            
            def portfolio_volatility(weights, mean_returns, cov_matrix):
                return portfolio_performance(weights, mean_returns, cov_matrix)[1]
            
            def negative_return(weights, mean_returns, cov_matrix):
                return -np.dot(weights, mean_returns)
            
            # Optimization setup
            num_assets = len(tickers)
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Run optimizations
            optimized = minimize(
                negative_sharpe,
                initial_weights,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            min_vol = minimize(
                portfolio_volatility,
                initial_weights,
                args=(mean_returns, cov_matrix),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            max_return = minimize(
                negative_return,
                initial_weights,
                args=(mean_returns, cov_matrix),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            # Select portfolio based on risk profile
            portfolio_map = {
                "Low": ("Minimum Volatility Portfolio", min_vol.x),
                "Medium": ("Maximum Sharpe Ratio Portfolio", optimized.x),
                "High": ("Maximum Return Portfolio", max_return.x)
            }
            
            portfolio_name, optimal_weights = portfolio_map[risk_profile]
            ret, risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            
            # Annualize metrics
            trading_days = 252
            annual_return = ret * trading_days
            annual_risk = risk * np.sqrt(trading_days)
            sharpe = annual_return / annual_risk
            
            # Wealth projection
            final_amount = investment * ((1 + annual_return) ** time_period)
            profit = final_amount - investment
            
            # Monte Carlo Simulation
            def monte_carlo_simulation(investment, mean_returns, cov_matrix, weights, years, simulations=10000):
                """Run Monte Carlo simulation for portfolio projections"""
                results = []
                
                for _ in range(simulations):
                    # Generate random returns based on mean and covariance
                    random_returns = np.random.multivariate_normal(
                        mean_returns * 252,  # Annualized
                        cov_matrix * 252,    # Annualized
                        years
                    )
                    
                    # Calculate portfolio returns for each year
                    portfolio_returns = np.dot(random_returns, weights)
                    
                    # Calculate final value
                    final_value = investment
                    for year_return in portfolio_returns:
                        final_value *= (1 + year_return)
                    
                    results.append(final_value)
                
                return np.array(results)
            
            # Run Monte Carlo
            mc_results = monte_carlo_simulation(
                investment, 
                mean_returns, 
                cov_matrix, 
                optimal_weights, 
                time_period,
                simulations=10000
            )
            
            # Calculate confidence intervals
            percentile_5 = np.percentile(mc_results, 5)
            percentile_25 = np.percentile(mc_results, 25)
            percentile_50 = np.percentile(mc_results, 50)
            percentile_75 = np.percentile(mc_results, 75)
            percentile_95 = np.percentile(mc_results, 95)
            
            # Probability of profit
            prob_profit = (mc_results > investment).sum() / len(mc_results) * 100
            prob_double = (mc_results > investment * 2).sum() / len(mc_results) * 100
            
            # Success message
            st.success(f"✅ Optimization complete! Selected: **{portfolio_name}**")
            
            # Key metrics in columns
            st.markdown("---")
            st.subheader("📊 Portfolio Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Annual Return",
                    f"{annual_return*100:.2f}%",
                    delta=f"{annual_return*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Annual Volatility",
                    f"{annual_risk*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    delta="Higher is better"
                )
            
            with col4:
                st.metric(
                    "Expected Final Value",
                    f"₹{final_amount:,.0f}",
                    delta=f"₹{profit:,.0f} profit",
                    help=f"90% confidence: ₹{percentile_5:,.0f} - ₹{percentile_95:,.0f}"
                )
            
            # Tabs for different views
            st.markdown("---")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Allocation", "💰 Investment Breakdown", "🎲 Monte Carlo Simulation", "📉 Historical Performance", "🎯 All Strategies"])
            
            with tab1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Pie chart
                    allocation_df = pd.DataFrame({
                        'Stock': tickers,
                        'Weight': optimal_weights * 100,
                        'Amount': optimal_weights * investment
                    }).sort_values('Weight', ascending=False)
                    
                    fig_pie = px.pie(
                        allocation_df,
                        values='Weight',
                        names='Stock',
                        title='Portfolio Allocation',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=500)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        allocation_df,
                        x='Stock',
                        y='Weight',
                        title='Weight by Stock',
                        color='Weight',
                        color_continuous_scale='Blues',
                        labels={'Weight': 'Allocation (%)'}
                    )
                    fig_bar.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
                st.subheader("💰 Investment Breakdown")
                
                # Create detailed allocation table
                allocation_detailed = pd.DataFrame({
                    'Stock': tickers,
                    'Weight (%)': optimal_weights * 100,
                    'Investment (₹)': optimal_weights * investment,
                    'Expected Return': mean_returns.values * trading_days * 100
                }).sort_values('Weight (%)', ascending=False)
                
                # Format the dataframe
                allocation_detailed['Weight (%)'] = allocation_detailed['Weight (%)'].map('{:.2f}%'.format)
                allocation_detailed['Investment (₹)'] = allocation_detailed['Investment (₹)'].map('₹{:,.2f}'.format)
                allocation_detailed['Expected Return'] = allocation_detailed['Expected Return'].map('{:.2f}%'.format)
                
                st.dataframe(
                    allocation_detailed,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Wealth projection
                st.subheader("📈 Wealth Projection")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Initial Investment:** ₹{investment:,.2f}  
                    **Investment Period:** {time_period} years  
                    **Expected Annual Return:** {annual_return*100:.2f}%
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Expected Final Value:** ₹{final_amount:,.2f}  
                    **Total Profit:** ₹{profit:,.2f}  
                    **Total Return:** {(profit/investment)*100:.2f}%
                    """)
            
            with tab3:
                st.subheader("🎲 Monte Carlo Simulation")
                st.markdown(f"**10,000 simulations** of possible portfolio outcomes over **{time_period} years**")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Distribution histogram
                    fig_mc = go.Figure()
                    
                    fig_mc.add_trace(go.Histogram(
                        x=mc_results,
                        nbinsx=50,
                        name='Simulated Outcomes',
                        marker_color='rgba(102, 126, 234, 0.7)',
                        hovertemplate='Value: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    # Add percentile lines
                    colors = ['red', 'orange', 'green', 'orange', 'red']
                    percentiles = [percentile_5, percentile_25, percentile_50, percentile_75, percentile_95]
                    labels = ['5th', '25th', '50th (Median)', '75th', '95th']
                    
                    for pct, label, color in zip(percentiles, labels, colors):
                        fig_mc.add_vline(
                            x=pct,
                            line_dash="dash",
                            line_color=color,
                            annotation_text=f"{label}: ₹{pct:,.0f}",
                            annotation_position="top"
                        )
                    
                    fig_mc.update_layout(
                        title='Distribution of Possible Outcomes',
                        xaxis_title='Final Portfolio Value (₹)',
                        yaxis_title='Frequency',
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Key Statistics")
                    
                    st.metric("Median Outcome", f"₹{percentile_50:,.0f}")
                    st.metric("Best Case (95th)", f"₹{percentile_95:,.0f}", 
                             delta=f"+{((percentile_95/investment - 1)*100):.1f}%")
                    st.metric("Worst Case (5th)", f"₹{percentile_5:,.0f}",
                             delta=f"{((percentile_5/investment - 1)*100):.1f}%")
                    
                    st.divider()
                    
                    st.markdown("### 🎯 Probabilities")
                    st.progress(prob_profit/100, text=f"**{prob_profit:.1f}%** chance of profit")
                    st.progress(prob_double/100, text=f"**{prob_double:.1f}%** chance of doubling")
                
                # Confidence intervals
                st.markdown("---")
                st.subheader("📈 Confidence Intervals")
                
                confidence_data = pd.DataFrame({
                    'Confidence Level': ['90% Confidence', '50% Confidence', 'Expected (Mean)'],
                    'Lower Bound': [
                        f"₹{percentile_5:,.0f}",
                        f"₹{percentile_25:,.0f}",
                        f"₹{mc_results.mean():,.0f}"
                    ],
                    'Upper Bound': [
                        f"₹{percentile_95:,.0f}",
                        f"₹{percentile_75:,.0f}",
                        f"₹{mc_results.mean():,.0f}"
                    ],
                    'Range': [
                        f"₹{percentile_95 - percentile_5:,.0f}",
                        f"₹{percentile_75 - percentile_25:,.0f}",
                        "—"
                    ]
                })
                
                st.dataframe(confidence_data, use_container_width=True, hide_index=True)
                
                st.info("""
                💡 **How to read this:**
                - **90% Confidence**: There's a 90% chance your final value will be between the lower and upper bounds
                - **50% Confidence**: There's a 50% chance your final value will be in this narrower range
                - **Median**: The middle value of all simulations (50th percentile)
                """)
            
            with tab4:
                st.subheader("📉 Historical Performance")
                
                # Calculate portfolio historical returns
                portfolio_returns = (returns * optimal_weights).sum(axis=1)
                cumulative_returns = (1 + portfolio_returns).cumprod()
                
                # Create line chart
                fig_historical = go.Figure()
                
                fig_historical.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#667eea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                
                fig_historical.update_layout(
                    title='Cumulative Returns (Based on Historical Data)',
                    xaxis_title='Date',
                    yaxis_title='Growth of ₹1',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_historical, use_container_width=True)
                
                # Performance statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_return = (cumulative_returns.iloc[-1] - 1) * 100
                    st.metric("Total Historical Return", f"{total_return:.2f}%")
                
                with col2:
                    max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                with col3:
                    best_day = portfolio_returns.max() * 100
                    st.metric("Best Daily Return", f"{best_day:.2f}%")
            
            with tab5:
                st.subheader("🎯 Comparison of All Strategies")
                
                # Calculate metrics for all strategies
                strategies = {
                    "Minimum Volatility": min_vol.x,
                    "Maximum Sharpe": optimized.x,
                    "Maximum Return": max_return.x
                }
                
                strategy_metrics = []
                for name, weights in strategies.items():
                    ret, risk = portfolio_performance(weights, mean_returns, cov_matrix)
                    ann_ret = ret * trading_days
                    ann_risk = risk * np.sqrt(trading_days)
                    sharpe_val = ann_ret / ann_risk
                    final_val = investment * ((1 + ann_ret) ** time_period)
                    
                    strategy_metrics.append({
                        'Strategy': name,
                        'Annual Return (%)': ann_ret * 100,
                        'Annual Risk (%)': ann_risk * 100,
                        'Sharpe Ratio': sharpe_val,
                        'Expected Final Value (₹)': final_val
                    })
                
                comparison_df = pd.DataFrame(strategy_metrics)
                
                # Format the dataframe
                st.dataframe(
                    comparison_df.style.format({
                        'Annual Return (%)': '{:.2f}',
                        'Annual Risk (%)': '{:.2f}',
                        'Sharpe Ratio': '{:.2f}',
                        'Expected Final Value (₹)': '₹{:,.0f}'
                    }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Risk-Return scatter plot
                fig_scatter = go.Figure()
                
                for name, weights in strategies.items():
                    ret, risk = portfolio_performance(weights, mean_returns, cov_matrix)
                    ann_ret = ret * trading_days * 100
                    ann_risk = risk * np.sqrt(trading_days) * 100
                    
                    color = '#00CC96' if name == portfolio_name.replace(' Portfolio', '') else '#636EFA'
                    size = 20 if name == portfolio_name.replace(' Portfolio', '') else 15
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=[ann_risk],
                        y=[ann_ret],
                        mode='markers+text',
                        name=name,
                        marker=dict(size=size, color=color),
                        text=[name],
                        textposition="top center"
                    ))
                
                fig_scatter.update_layout(
                    title='Risk-Return Profile of All Strategies',
                    xaxis_title='Annual Risk (Volatility) %',
                    yaxis_title='Annual Return %',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Disclaimer
            st.markdown("---")
            st.info("""
            ⚠️ **Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results. 
            Projections are based on historical data and should not be considered financial advice. 
            Please consult with a qualified financial advisor before making investment decisions.
            """)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to the Smart Portfolio Optimizer!
    
    This tool helps you build an optimized investment portfolio using **Modern Portfolio Theory**. 
    
    #### How it works:
    1. 💰 Set your investment amount
    2. 🎯 Choose your risk appetite (Low/Medium/High)
    3. 📅 Select your investment horizon
    4. 🚀 Click "Optimize Portfolio" to get your personalized allocation
    
    #### What you'll get:
    - 📊 Optimal portfolio allocation across selected stocks
    - 📈 Expected returns and risk metrics
    - 💰 Detailed investment breakdown
    - 📉 Historical performance analysis
    - 🎯 Comparison of different strategies
    
    ---
    
    **Get started by configuring your preferences in the sidebar!** 👈
    """)
    
    # Sample visualization
    st.subheader("📊 Sample Portfolio Allocation")
    sample_data = pd.DataFrame({
        'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'Other'],
        'Allocation': [25, 20, 18, 15, 22]
    })
    
    fig = px.pie(
        sample_data,
        values='Allocation',
        names='Stock',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("This is just a sample. Your actual allocation will be optimized based on your inputs.")