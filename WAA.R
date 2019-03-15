WAA = function(outcomes, X, q, C) {
  #function implements Weak Aggregating Algorithm (WAA)
  #Inputs: outcomes - vector of outcomes of length T,
  #        X - matrix of features where rows are observations and columns are experts [T x N],
  #        q - chosen quantile
  #        C - positive constant (parameter of WAA)
  
  #Outputs: gamma - calculated predictions of WAA,
  #         weights_norm - normalised weights of experts for each time step
  
  calc_loss = function(outcomes, predictions, q) {
    #function calculates pinball loss for vectors outcomes
    #and predictions for quantile q
    losses = outcomes - predictions
    for (j in 1:length(losses)) {
      losses[j] = ifelse(losses[j] > 0, q*losses[j], (q-1)*losses[j])
    }
    return(losses)
  }
  T = dim(X)[1]  #time
  N = dim(X)[2]  #number of experts
  weights_init = rep(1 / N, N)    #equal initial weights
  weights_norm = matrix(0, nrow = T, ncol = N)
  gamma = matrix(0, nrow = T, ncol = 1)
  losses_tot = 0
  for (t in 1:T) {
    weights = weights_init * exp(-C * losses_tot / sqrt(t))  #update weights
    weights_norm[t, ] = weights / sum(weights)               #normalise weights
    gamma[t, ] = sum(X[t, ] * weights_norm[t, ])             #prediction of WAA 
    losses = calc_loss(outcomes[t], X[t, ], q)     #current losses of experts
    losses_tot = losses_tot + losses                         #update total losses
  }
  return(list(gamma = gamma, weights_norm = weights_norm))
}
