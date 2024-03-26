# Farthest Point Sampling in Chemical Feature Space

Our research introduces the farthest point sampling (FPS) strategy within targeted chemical feature spaces to generate well-distributed training datasets. This approach enhances model performance by increasing the diversity within the training data's chemical feature space. We rigorously evaluated this strategy across various ML models – including artificial neural networks (ANN), support vector machines (SVM), random forests (RF) etc. – using datasets encapsulating key physicochemical properties. Our findings demonstrate that FPS-based models markedly outperform those trained via random sampling in terms of predictive accuracy, robustness, and a notable reduction in overfitting, especially in smaller training datasets.

![Fig 1](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig1.png)

A graphic illustration of the farthest point sampling in chemical space

![Fig 2](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig2.png)

MSE compared between FPS and RS

![Fig 3](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig3.png)

MSE compared by sampling in different chemical space

![Fig 4](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig4.png)

Heatmap of MSE for different machine learning model

![Fig 5](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig5.png)

MSE for different physicochemical datasets

![Fig 6](https://github.com/yuxi-TJU/Farthest-Point-Sampling-in-Chemical-Feature-Space/blob/main/figs/fig6.png)

t-SNE distributions for FPS and RS
