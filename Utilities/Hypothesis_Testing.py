import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
#file_path = r"C:\Users\jacky\Desktop\Research Project\Code Repo\scheduling_results_5.csv"
# file_path = r"C:\Users\jacky\Desktop\Research Project\Code Repo\scheduling_results_10.csv"
# file_path = r"C:\Users\jacky\Desktop\Research Project\Code Repo\scheduling_results_15.csv"
file_path = r"C:\Users\jacky\Desktop\Research Project\Code Repo\scheduling_results_20.csv"
#file_path = r"C:\Users\jacky\Desktop\Research Project\Code Repo\scheduling_results_25.csv"

all_results = pd.read_csv(file_path)

# Check the column names to ensure they are correct
print("Column names in the dataset:", all_results.columns)

# Separate data based on strategy and use the correct column
if 'Average Resource Utilization Std Dev' in all_results.columns:
    lrp_std_dev = all_results[all_results['Strategy'] == 'Least Requested Priority']['Average Resource Utilization Std Dev']
    bra_std_dev = all_results[all_results['Strategy'] == 'Balanced Resource Allocation']['Average Resource Utilization Std Dev']
    wrr_std_dev = all_results[all_results['Strategy'] == 'Weighted Residual Resource']['Average Resource Utilization Std Dev']
else:
    raise KeyError("The column 'Average Resource Utilization Std Dev' does not exist in the dataset.")

# Perform hypothesis testing
# Shapiro-Wilk test for normality
shapiro_lrp = stats.shapiro(lrp_std_dev)
shapiro_bra = stats.shapiro(bra_std_dev)
shapiro_wrr = stats.shapiro(wrr_std_dev)

print(f"Shapiro-Wilk Test for LRP: {shapiro_lrp}")
print(f"Shapiro-Wilk Test for BRA: {shapiro_bra}")
print(f"Shapiro-Wilk Test for WRR: {shapiro_wrr}")

# Q-Q plots to confirm normality
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
stats.probplot(lrp_std_dev, dist="norm", plot=plt)
plt.title("Q-Q Plot for LRP", fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=14)
plt.ylabel('Sample Quantiles', fontsize=14)

plt.subplot(1, 3, 2)
stats.probplot(bra_std_dev, dist="norm", plot=plt)
plt.title("Q-Q Plot for BRA", fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=14)
plt.ylabel('Sample Quantiles', fontsize=14)

plt.subplot(1, 3, 3)
stats.probplot(wrr_std_dev, dist="norm", plot=plt)
plt.title("Q-Q Plot for WRR", fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=14)
plt.ylabel('Sample Quantiles', fontsize=14)

plt.tight_layout()
plt.show()

# Function to visualize p-value with normal distribution
def plot_wilcoxon_p_value(stat, n, title):
    mu_w = n * (n + 1) / 4
    sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    
    x = np.linspace(mu_w - 4*sigma_w, mu_w + 4*sigma_w, 1000)
    y = stats.norm.pdf(x, mu_w, sigma_w)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Normal Distribution of W', color='blue')
    plt.axvline(stat, color='red', linestyle='--', label=f'W Test Statistic = {stat:.2f}')
    
    # Significance level (alpha = 0.05)
    critical_value = stats.norm.ppf(0.05, mu_w, sigma_w)
    plt.axvline(critical_value, color='green', linestyle='--', label=f'Significance Level (0.05) = {critical_value:.2f}')
    
    # Shade the p-value area
    if stat < mu_w:
        x_fill = np.linspace(mu_w - 4*sigma_w, stat, 1000)
        plt.fill_between(x_fill, stats.norm.pdf(x_fill, mu_w, sigma_w), color='red', alpha=0.3, label=f'p-value = {stats.norm.cdf(stat, mu_w, sigma_w):.2e}')
    else:
        x_fill = np.linspace(stat, mu_w + 4*sigma_w, 1000)
        plt.fill_between(x_fill, stats.norm.pdf(x_fill, mu_w, sigma_w), color='red', alpha=0.3, label=f'p-value = {1 - stats.norm.cdf(stat, mu_w, sigma_w):.2e}')
    
    plt.title(title, fontsize=18)
    plt.xlabel('W', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend()
    plt.show()

# Choose t-Test or Wilcoxon based on normality
if shapiro_lrp.pvalue > 0.05 and shapiro_bra.pvalue > 0.05 and shapiro_wrr.pvalue > 0.05:
    # t-Test
    t_test_wrr_lrp = stats.ttest_ind(wrr_std_dev, lrp_std_dev, alternative='less')
    t_test_wrr_bra = stats.ttest_ind(wrr_std_dev, bra_std_dev, alternative='less')

    print(f"t-Test between WRR and LRP: {t_test_wrr_lrp}")
    print(f"t-Test between WRR and BRA: {t_test_wrr_bra}")
else:
    # Wilcoxon test
    wilcoxon_wrr_lrp = stats.wilcoxon(wrr_std_dev, lrp_std_dev, alternative='less')
    wilcoxon_wrr_bra = stats.wilcoxon(wrr_std_dev, bra_std_dev, alternative='less')

    print(f"Wilcoxon test between WRR and LRP: {wilcoxon_wrr_lrp}")
    print(f"Wilcoxon test between WRR and BRA: {wilcoxon_wrr_bra}")

    # Plot p-values with normal distribution of W
    n = len(wrr_std_dev)
    plot_wilcoxon_p_value(wilcoxon_wrr_lrp.statistic, n, "Wilcoxon Test Statistic (WRR vs LRP)")
    plot_wilcoxon_p_value(wilcoxon_wrr_bra.statistic, n, "Wilcoxon Test Statistic (WRR vs BRA)")
