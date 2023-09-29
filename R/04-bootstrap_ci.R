# Calcolo degli intervalli di confidenza per l'AUC utilizzando il bootstrapping

bootstrap_auc <- function(auc_vec, n_bootstraps = 1000, alpha = 0.05) {
  bootstrapped_auc <- numeric(n_bootstraps)
  n = length(auc_vec)
  
  for(i in 1:n_bootstraps) {
    # Eseguire il campionamento con sostituzione
    resampled_indices <- sample(1:n, n, replace = TRUE)
    bootstrapped_auc[i] <- mean(auc_vec[resampled_indices])
  }
  
  # Calcolare gli intervalli di confidenza
  ci_lower <- quantile(bootstrapped_auc, alpha / 2)
  ci_upper <- quantile(bootstrapped_auc, 1 - alpha / 2)
  
  return(data.frame(CI_lower = ci_lower, CI_upper = ci_upper))
}

# Esempio di calcolo degli IC per le AUC di ogni modello
# Supponendo che auc_lr, auc_nplr_spline, auc_nplr_kernel, ecc.
# siano i vettori delle AUC per ogni fold di cross-validazione per ogni metodo

ic_auc_lr <- bootstrap_auc(auc_lr)
ic_auc_nplr_spline <- bootstrap_auc(auc_nplr_spline)
ic_auc_nplr_kernel <- bootstrap_auc(auc_nplr_kernel)
# ... fare lo stesso per gli altri metodi


