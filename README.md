# Predicting the Whiff: A Pitch-Level Machine Learning Analysis

</a>

In modern baseball, the ability to miss bats is the ultimate currency of elite pitching. In 2025, swing-and-miss ability continues to be the strongest predictor of run prevention because whiffs eliminate the volatility of defensive positioning and "babip" luck. However, evaluating a pitcher solely on aggregate strikeout totals can be misleading. Pitchers are often judged by their results, but front offices need to know if those results are a product of sustainable "stuff" or a fortunate sequence of hitter approach and location.

This forms the basis of why I conducted the analysis detailed in this project: I wanted to know if a whiff could be predicted as a probabilistic event based solely on a pitch's physical profile. Using pitch-level Statcast data, I built a machine learning framework to estimate the likelihood of a swing-and-miss for every individual throw. While velocity and movement are well-known drivers of success, the inherent noise of human reaction time makes pitch-level prediction a complex classification problem that requires more than just raw accuracy to solve.

This project set out to answer three key questions using Python, PyTorch, and Statcast tracking data:

* Can I use machine learning to predict likelihood of a whiff only using pitch charectestics given by Statcast?
* Can a non-linear Multi-Layer Perceptron (MLP) significantly outperform a linear Logistic Regression in capturing the "stuff" of a pitch?
* How critical is post-hoc calibration (Platt Scaling) in turning raw model outputs into reliable, "honest" probabilities?

By breaking down individual pitch characteristics and applying advanced modeling techniques, we can better understand how specific physical traits shape the game and provide a more objective metric for player evaluation and pitch design.
