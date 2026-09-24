// Mixed-effects Bayesian normal regression for longitudinal continuous outcomes.
// Models Day 90 and Week 36 symptom burden scores jointly.
//
// y_it ~ N(mu_it, sigma_y^2)
// mu_it = alpha_i + x_i' * bbeta + theta * z_i
//         + delta * t_week36[i] + xi * z_i * t_week36[i]
//
// alpha_i ~ N(alpha0, sigma_a^2)   patient random intercept (non-centred)
//
// Priors (revision 9.1, outcome is standardised z-score):
//   alpha0                    ~ N(alpha_loc=0, 1^2)
//   theta, delta, xi, bbeta_k ~ N(0, 0.5^2)
//   sigma_y                   ~ Half-Normal(0, 1^2)
//   sigma_a                   ~ Half-Normal(0, 1^2)

data {
  int<lower=1> N;
  int<lower=1> Np;
  int<lower=1> M;
  vector[N] y;
  array[N] int<lower=1, upper=Np> pid;
  vector[N] z;
  vector[N] t_week36;
  matrix[N, M] X;
  real alpha_loc;
}
parameters {
  real alpha0;
  vector[Np] alpha_raw;
  real<lower=0> sigma_a;
  real theta;                              // treatment effect at Day 90
  real delta;                              // time main effect (Week 36 vs Day 90)
  real xi;                                 // treatment x time interaction
  vector[M] bbeta;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[Np] alpha;
  vector[N]  mu;
  alpha = alpha0 + sigma_a * alpha_raw;
  for (i in 1:N)
    mu[i] = alpha[pid[i]] + theta * z[i] + delta * t_week36[i]
            + xi * z[i] * t_week36[i] + X[i] * bbeta;
}
model {
  alpha0    ~ normal(alpha_loc, 1);
  alpha_raw ~ std_normal();
  sigma_a   ~ normal(0, 1);
  theta     ~ normal(0, 0.5);
  delta     ~ normal(0, 0.5);
  xi        ~ normal(0, 0.5);
  bbeta     ~ normal(0, 0.5);
  sigma_y   ~ normal(0, 1);
  y ~ normal(mu, sigma_y);
}
