"""
Noiseless simulation using stan including function that is evaluated within a
system of ordinary differential equations.

The model simulates growth on glcuose with a biomass yield of 0.09 gDW/mol_glc
and a maximum specific growth rate of 0.22 1/h.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan



###############################################################################
# INPUTS

var_dict = {
        'p[1]': 'mu_max',
        'p[2]': 'Yxs',
        }

stan_str = """
functions {
    real fglc (
        real[] y
    )
    {
        real vglc;
        vglc = (y[1]/1) / (1 + (y[1]/1));
        return vglc;
    }

    real[] myodes(
        real t,
        real[] y,
        real[] p,
        real[] x_r,
        int[] x_i
        )
        {
            real dydt[2];
            dydt[1] = -y[2] * (0.22 / 0.09) * fglc(y);
            dydt[2] = (y[2] * 0.22 * fglc(y));
            return dydt;
        }
}

data {
    int<lower=1> T;
    real t0;
    real t_sim[T];
    real<lower=0> y0[2]; // init
    }

transformed data {
    real x_r[0];
    int x_i[0];
}

parameters {
    vector<lower=0>[2] sigma;
    real<lower=0> p[2];
}

transformed parameters {
    real y_hat[T, 2];
    y_hat = integrate_ode_rk45(myodes, y0, t0, t_sim, p, x_r, x_i);
}

model {
}

generated quantities {
}
"""
###############################################################################
# STATEMENTS

# Dictionary to pass to the stan model
t_sim = np.arange(1, 10, 0.5)
datadict = {}
datadict['t0'] = 0
datadict['t_sim'] = t_sim
datadict['T'] = len(t_sim)
datadict['y0'] = [100, 0.2]

# compile StanModel
sm = pystan.StanModel(model_code=stan_str)

fit = sm.sampling(
                    data=datadict,
                    iter=1,
                    chains=1,
                    n_jobs=1,
                    algorithm='Fixed_param',
                    seed = 42
                    )

summary = fit.summary()
df_summary =  pd.DataFrame(
    summary['summary'],
    columns=summary['summary_colnames'],
    index=summary['summary_rownames']
)

fit_dict= fit.extract()
fit_df = pd.DataFrame(fit_dict['y_hat'][0], index=t_sim, columns=['glc', 'dw'])

fit_df.plot(subplots=True)
plt.show()
