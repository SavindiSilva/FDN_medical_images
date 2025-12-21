import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture

class Sieve:
    def __init__(self, output_dir="experiments/sieve_plots"):
        self.gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(self, losses, epoch):
        """
        Fits GMM and plots the histogram.
        Does NOT return probabilities for training (we don't need them yet).
        """
        losses = np.array(losses).reshape(-1, 1)
        
        #normalize loses
        min_l, max_l = losses.min(), losses.max()
        losses_norm = (losses - min_l) / (max_l - min_l + 1e-8)
        
        #fit gmm
        self.gmm.fit(losses_norm)
        
        #identify clusters
        means = self.gmm.means_.flatten()
        clean_idx = np.argmin(means) 
        probs = self.gmm.predict_proba(losses_norm)
        clean_probs = probs[:, clean_idx]
        
        #visulalization
        self._plot_histogram(losses, clean_probs, epoch)
        
        print(f"[Sieve Analysis] plot saved to {self.output_dir}")

    def _plot_histogram(self, losses, clean_probs, epoch):
        plt.figure(figsize=(10, 6))
        
        #plot distribution
        plt.hist(losses, bins=100, alpha=0.5, label='All Samples', color='gray')
        
        #overlay "clean" prediction
        clean_losses = np.array(losses)[clean_probs > 0.5]
        plt.hist(clean_losses, bins=100, alpha=0.5, color='green', label='Identified Clean')
        
        plt.title(f"Sieve Diagnostic: Loss Distribution at Epoch {epoch}")
        plt.xlabel("Loss Value (Hardness)")
        plt.ylabel("Count")
        plt.legend()
        
        plt.savefig(f"{self.output_dir}/diagnostic_epoch_{epoch}.png")
        plt.close()