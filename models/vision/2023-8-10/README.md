# Model Info

This uses the AutolandPerceptionModel with ResNet 50. It was trained until convergence with the ResNet backbone frozen. After this, we trained it further while allowing the backbone to update.

There were 16,359 unique data points in this dataset from ground range 0m to ~9,000m. Note that we need more data up until ~12,400m for the whole glideslope.

The final observed MSE was 0.000123. Since MSE is 0.5(x^2 + y^2) and x and y are both in km, this is approximately sqrt(0.000123)=0.01109km=11.09m error per dimension.
