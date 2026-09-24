// Mixed-effects Bayesian logistic regression for longitudinal binary outcomes.
// Models Day 90 and Week 36 observations jointly with patient random intercepts.
//
// Linear predictor (logit link):
//   g(mu_it) = alpha_i + x_i' * bbeta + gamma * z_i
//              + delta * t_week36[i] + zeta * z_i * t_week36[i]
//
// alpha_i ~ N(alpha0, sigma_a^2)   patient random intercept (non-centred)
// z_i      treatment indicator (1 = Paxlovid)
// t_week36 time indicator (1 = Week 36, 0 = Day 90)
//
// Priors (revision 9):
//   alpha0          ~ N(alpha_loc, 2.5^2)
//   gamma, delta, zeta, bbeta_k ~ N(0, 2.5^2)
//   sigma_a         ~ Half-Normal(0, 2.5^2)

data {
  int<lower=1> N;                          // total observations (long format)
  int<lower=1> Np;                         // number of patients
  int<lower=1> M;                          // number of fixed covariates
  array[N] int<lower=0, upper=1> y;        // binary outcome
  array[N] int<lower=1, upper=Np> pid;     // patient index (1-based)
  vector[N] z;                             // treatment indicator
  vector[N] t_week36;                      // time indicator (1 = Week 36)
  matrix[N, M] X;                          // covariates (pre-standardised)
  real alpha_loc;                          // intercept prior location
}
parameters {
  real alpha0;                             // population-level intercept
  vector[Np] alpha_raw;                    // non-centred random intercepts
  real<lower=0> sigma_a;                   // SD of random intercepts
  real gamma;                              // treatment effect (log-OR at Day 90)
  real delta;                              // time main effect (Week 36 vs Day 90)
  real zeta;                               // treatment x time interaction
  vector[M] bbeta;                         // covariate effects
}
transformed parameters {
  vector[Np] alpha;
  vector[N]  mu;
  alpha = alpha0 + sigma_a * alpha_raw;    // non-centred parameterisation
  for (i in 1:N)
    mu[i] = alpha[pid[i]] + gamma * z[i] + delta * t_week36[i]
            + zeta * z[i] * t_week36[i] + X[i] * bbeta;
}
model {
  alpha0    ~ normal(alpha_loc, 2.5);
  alpha_raw ~ std_normal();
  sigma_a   ~ normal(0, 2.5);
  gamma     ~ normal(0, 2.5);
  delta     ~ normal(0, 2.5);
  zeta      ~ normal(0, 2.5);
  bbeta     ~ normal(0, 2.5);
  y ~ bernoulli_logit(mu);
}
