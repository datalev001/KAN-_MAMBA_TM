# Advancing Time Series Forecasting: A Comparative Study of Mamba, GRU, KAN, GNN, and ARMA Models
Evaluating Modern and Traditional Methods for Multivariate Time Series Prediction
Exploring Dynamic Weighting, State-Space Modeling, and ARMA to Compare Strengths, Address Challenges, and Deliver Superior Forecasting Performance
Imagine trying to predict how a group of related factors change over time - like stock prices that affect each other, gene activity in a biological system, or sales of connected products in a supply chain. These aren't just separate time series; they interact and influence one another in ways traditional models often miss. This is what makes multivariate time series forecasting both fascinating and challenging.
For years, models like ARMA have been the standard for time series forecasting. They work well for handling one series at a time but struggle to capture relationships between multiple series. In reality, that's a big gap. Stock prices are connected, economic indicators move together, and the sales of one product can impact the entire inventory. What we need are models that not only track individual trends but also understand how these series influence each other.
In this paper, I explore five different approaches for tackling multivariate forecasting:
The Mamba-inspired model for capturing time-based dynamics.
GRU (Gated Recurrent Units) for finding patterns in sequences.
Kolmogorov-Arnold Networks (KAN) with dynamic weights to uncover nonlinear relationships.
Graph Neural Networks (GNNs) to map relationships between series as a graph.
ARMA, a classic model used as a baseline.

By testing these methods individually, I aim to show how each one performs, their strengths, and where they struggle when dealing with multiple, interconnected series.
This isn't just about theory. It's about understanding what tools we have and how they can help in fields where connections matter - whether it's predicting stock market trends, optimizing supply chains, or understanding complex biological systems.
